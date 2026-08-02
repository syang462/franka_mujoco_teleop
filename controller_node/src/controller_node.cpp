// Controller node — no MuJoCo, no rendering. It only:
//   - subscribes to the robot state the sim/hardware node publishes:
//       /robot/joint_states     (sensor_msgs/JointState: q, qdot)
//       /robot/ee_pose          (geometry_msgs/PoseStamped)
//       /robot/jacobian         (std_msgs/Float64MultiArray, 6x7 row-major)
//       /robot/external_wrench  (geometry_msgs/WrenchStamped)
//   - subscribes to the haptic device input (/haptic/pose, /haptic/buttons
//     handled on the hardware side, /toggle_mode)
//   - publishes the control law's output joint-velocity command to
//       /target_joint_velocities (std_msgs/Float64MultiArray)
//   - publishes the force to render back on the haptic device to
//       /haptic/force_command
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <Eigen/Dense>

const int N_ARM_JOINTS = 7;

// Snapshot of everything the control law needs, assembled from three
// independently-arriving topics.
struct RobotState {
    bool valid = false;
    Eigen::VectorXd q{Eigen::VectorXd::Zero(N_ARM_JOINTS)};
    Eigen::VectorXd qdot{Eigen::VectorXd::Zero(N_ARM_JOINTS)};
    Eigen::MatrixXd jacobian{Eigen::MatrixXd::Zero(6, N_ARM_JOINTS)};
    Eigen::VectorXd external_force{Eigen::VectorXd::Zero(6)};
    Eigen::Vector3d ee_pos{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond ee_orientation{Eigen::Quaterniond::Identity()};
};

class HapticControllerNode : public rclcpp::Node
{
public:
    HapticControllerNode()
    : Node("haptic_controller_node")
    {
        last_time_ = this->now();

        this->declare_parameter("use_topic_input", false);
        use_topic_input_ = this->get_parameter("use_topic_input").as_bool();
        if (use_topic_input_) {
            RCLCPP_INFO(this->get_logger(),
                "Topic input enabled: target from /target_pose, force from /robot_force");
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1),
            std::bind(&HapticControllerNode::controlLoop, this));

        // ---- Robot state subscriptions (the ONLY link to the sim/hardware node) ----
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/robot/joint_states", 1,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                if (static_cast<int>(msg->position.size()) != N_ARM_JOINTS ||
                    static_cast<int>(msg->velocity.size()) != N_ARM_JOINTS) {
                    return;
                }
                std::lock_guard<std::mutex> lock(state_mutex_);
                for (int i = 0; i < N_ARM_JOINTS; ++i) {
                    state_.q(i)    = msg->position[i];
                    state_.qdot(i) = msg->velocity[i];
                }
                have_joint_state_ = true;
                updateValidity();
            });

        ee_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/robot/ee_pose", 1,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                state_.ee_pos << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
                state_.ee_orientation = Eigen::Quaterniond(
                    msg->pose.orientation.w, msg->pose.orientation.x,
                    msg->pose.orientation.y, msg->pose.orientation.z);
                have_ee_pose_ = true;
                updateValidity();
            });

        jacobian_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/robot/jacobian", 1,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (static_cast<int>(msg->data.size()) != 6 * N_ARM_JOINTS) {
                    return;
                }
                std::lock_guard<std::mutex> lock(state_mutex_);
                for (int r = 0; r < 6; ++r) {
                    for (int c = 0; c < N_ARM_JOINTS; ++c) {
                        state_.jacobian(r, c) = msg->data[r * N_ARM_JOINTS + c];
                    }
                }
                have_jacobian_ = true;
                updateValidity();
            });

        wrench_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            "/robot/external_wrench", 1,
            [this](const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                state_.external_force << msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z,
                                          msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z;
                have_wrench_ = true;
                updateValidity();
            });

        // ---- Haptic device input ----
        haptic_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "haptic/pose", 1,
            std::bind(&HapticControllerNode::hapticPoseCallback, this, std::placeholders::_1));

        mode_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/toggle_mode", 10,
            [this](const std_msgs::msg::Int32::SharedPtr msg) {
                mode_ = (mode_ == 0) ? 1 : 0;
                RCLCPP_INFO(this->get_logger(), "Mode toggled -> %d (%s)",
                    mode_, mode_ == 0 ? "HOME pose" : "Haptic follow");
            });

        stiffness_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/stiffness", 10,
            std::bind(&HapticControllerNode::stiffnessCallback, this, std::placeholders::_1));

        // ---- Optional topic-driven input (use_topic_input:=true) ----
        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target_pose", 1,
            std::bind(&HapticControllerNode::targetPoseCallback, this, std::placeholders::_1));

        robot_force_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/robot_force", 1,
            std::bind(&HapticControllerNode::robotForceCallback, this, std::placeholders::_1));

        // ---- Outputs ----
        target_vel_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/target_joint_velocities", 1);
        haptic_force_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
            "/haptic/force_command", 1);
    }

private:

    void updateValidity() {
        // Called with state_mutex_ already held.
        state_.valid = have_joint_state_ && have_ee_pose_ && have_jacobian_ && have_wrench_;
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time last_time_;

    double m_ = 10;
    double zeta = 1.5;
    double k_ = 200; //100 -> 2000
    double d_ = 2.0 * zeta * std::sqrt(m_ * k_);
    int mode_ = 0;
    bool use_topic_input_ = false;

    Eigen::VectorXd home_q{{0.0, 0.0, 0.0, -M_PI/2.0, 0.0, M_PI/2.0, M_PI/4.0}};

    Eigen::Vector3d target_pose{0.5, 0.0, 0.25};
    Eigen::Quaterniond target_orientation{0.0, 1.0, 0.0, 0.0};
    Eigen::Vector3d ee_vel_ = Eigen::Vector3d::Zero();

    double position_scale_x_ = 4.0;
    double position_scale_y_ = 4.0;
    double position_scale_z_ = 4.0;
    double height_offset_ = 0.3;
    double max_force_output_ = 5.0;

    Eigen::Vector3d robot_force_ = Eigen::Vector3d::Zero();

    bool inside_workspace = false;
    bool nearSwitch = false;
    geometry_msgs::msg::Vector3 wall_force;

    // Robot state, assembled from topics
    std::mutex state_mutex_;
    RobotState state_;
    bool have_joint_state_ = false;
    bool have_ee_pose_ = false;
    bool have_jacobian_ = false;
    bool have_wrench_ = false;

    bool reachedStart = false;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr jacobian_sub_;
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr haptic_pose_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr mode_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr stiffness_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr robot_force_sub_;


    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_vel_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr haptic_force_pub_;

    void stiffnessCallback(const std_msgs::msg::Float64::SharedPtr msg) {
        k_ = msg->data;
        d_ = 2.0 * zeta * std::sqrt(m_ * k_);
        RCLCPP_INFO(this->get_logger(), "Stiffness updated: K = %.2f, D (calculated) = %.2f", k_, d_);
    }

    void hapticPoseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
        if (use_topic_input_) {
            return;
        }
        target_pose.x() = (-msg->position.z * position_scale_x_) + 0.35;
        target_pose.y() = -msg->position.x * position_scale_y_;
        target_pose.z() = (msg->position.y * position_scale_z_) + height_offset_;
    
        bool nearSwitchLocal = false;
        if (msg->position.y < -0.05 && msg->position.z < -0.05 && std::abs(msg->position.x) < 0.01) {
            nearSwitchLocal = true;
        }
        nearSwitch = nearSwitchLocal;

        Eigen::Quaterniond haptic_orientation(
            msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);

        const double x_min=0.2, x_max=0.9;
        const double y_min=-0.25, y_max=0.25;
        const double z_min=0.03, z_max=0.70;

        inside_workspace =
            (target_pose.x() >= x_min && target_pose.x() <= x_max) &&
            (target_pose.y() >= y_min && target_pose.y() <= y_max) &&
            (target_pose.z() >= z_min && target_pose.z() <= z_max);

        if (!nearSwitch) {
            const double k_wall = 100.0;
            Eigen::Vector3d boundary_force = Eigen::Vector3d::Zero();

            if      (target_pose.x() < x_min) boundary_force.x() =  k_wall * (x_min - target_pose.x());
            else if (target_pose.x() > x_max) boundary_force.x() = -k_wall * (target_pose.x() - x_max);

            if      (target_pose.y() < y_min) boundary_force.y() =  k_wall * (y_min - target_pose.y());
            else if (target_pose.y() > y_max) boundary_force.y() = -k_wall * (target_pose.y() - y_max);

            if      (target_pose.z() < z_min) boundary_force.z() =  k_wall * (z_min - target_pose.z());
            else if (target_pose.z() > z_max) boundary_force.z() = -k_wall * (target_pose.z() - z_max);

            if (boundary_force.norm() > 0.0) {
                geometry_msgs::msg::Vector3 force_out;
                force_out.x = -boundary_force.y();
                force_out.y =  boundary_force.z();
                force_out.z = -boundary_force.x();

                double mag = std::sqrt(force_out.x*force_out.x +
                                    force_out.y*force_out.y +
                                    force_out.z*force_out.z);
                if (mag > max_force_output_) {
                    double scale = max_force_output_ / mag;
                    force_out.x *= scale;
                    force_out.y *= scale;
                    force_out.z *= scale;
                }
                wall_force = force_out;
            }
        }

        target_pose.x() = std::clamp(target_pose.x(), x_min, x_max);
        target_pose.y() = std::clamp(target_pose.y(), y_min, y_max);
        target_pose.z() = std::clamp(target_pose.z(), z_min, z_max) + 0.1;

        Eigen::Quaterniond rot_1(Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond rot_0(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond base_down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
        target_orientation = (rot_1 * haptic_orientation * rot_0 * base_down).normalized();
    }

    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        if (!use_topic_input_) {
            return;
        }
        target_pose << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
        target_orientation = Eigen::Quaterniond(
            msg->pose.orientation.w, msg->pose.orientation.x,
            msg->pose.orientation.y, msg->pose.orientation.z);
    }

    void robotForceCallback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
        if (!use_topic_input_) {
            return;
        }
        robot_force_ << msg->x, msg->y, msg->z;
    }

    void robotForceToHapticForce(Eigen::VectorXd F_ext)
    {

        if (use_topic_input_) {
            return;
        }

        F_ext *= 0.25;//0.25; //0.3

        Eigen::Vector3d robot_force_in(-F_ext.x(), -F_ext.y(), -F_ext.z());

        robot_force_ = robot_force_in;

        geometry_msgs::msg::Vector3 force_out;
        force_out.x = -robot_force_in.y();
        force_out.y =  robot_force_in.z();
        force_out.z = -robot_force_in.x();

        const bool USE_LOW_PASS_FILTER  = true;
        const bool FORCE_OUTPUT_ENABLED = true;

        static geometry_msgs::msg::Vector3 filtered_force;
        static bool initialized = false;
        if (USE_LOW_PASS_FILTER) {
            const double alpha = 0.01;
            if (!initialized) { filtered_force = force_out; initialized = true; }
            else {
                filtered_force.x = alpha * force_out.x + (1.0 - alpha) * filtered_force.x;
                filtered_force.y = alpha * force_out.y + (1.0 - alpha) * filtered_force.y;
                filtered_force.z = alpha * force_out.z + (1.0 - alpha) * filtered_force.z;
            }
            force_out = filtered_force;
        }

        double magnitude = std::sqrt(
            force_out.x * force_out.x +
            force_out.y * force_out.y +
            force_out.z * force_out.z);
        if (magnitude > max_force_output_) {
            double scale = max_force_output_ / magnitude;
            force_out.x *= scale; force_out.y *= scale; force_out.z *= scale;
        }

        if (!FORCE_OUTPUT_ENABLED || nearSwitch) force_out.x = force_out.y = force_out.z = 0.0;

        haptic_force_pub_->publish(force_out);
    }

    void publishTargetVelocities(const Eigen::VectorXd& q_dot) {
        std_msgs::msg::Float64MultiArray msg;
        msg.data.resize(N_ARM_JOINTS);
        for (int i = 0; i < N_ARM_JOINTS; ++i) {
            msg.data[i] = q_dot(i);
        }
        target_vel_pub_->publish(msg);
    }

    void controlLoop() {
        auto now = this->now();
        double dt = (now - last_time_).seconds();
        last_time_ = now;
        if (dt <= 0.0 || dt > 0.1) {
            return;
        }

        RobotState state;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state = state_;
        }

        if (!state.valid) {
            //RCLCPP_WARN(this->get_logger(), "No robot state yet, skipping control tick");
            return;
        }

        Eigen::VectorXd q_dot(N_ARM_JOINTS);

        if (mode_ == 0) {
            const double k_home = 2.0;
            const double max_home_vel = 0.5;

            reachedStart = false;
            Eigen::VectorXd target_qdot_m1(N_ARM_JOINTS);
            for (int i = 0; i < N_ARM_JOINTS; i++) {
                double err = home_q[i] - state.q(i);
                target_qdot_m1(i) = std::clamp(k_home * err, -max_home_vel, max_home_vel);
            }

            q_dot = target_qdot_m1 - state.qdot;
            publishTargetVelocities(q_dot);
            return;
        }

        Eigen::Vector3d error = state.ee_pos - target_pose;
        
        //RCLCPP_INFO(this->get_logger(), "Current: [%.4f, %.4f, %.4f] | Target: [%.4f, %.4f, %.4f] | Error: [%.4f, %.4f, %.4f]",  state.ee_pos.x(), state.ee_pos.y(), state.ee_pos.z(),target_pose.x(), target_pose.y(), target_pose.z(), error.x(), error.y(), error.z());

        Eigen::Vector3d effective_force;
        
        bool readForce = true;
        if (readForce){
            effective_force = 0.5*robot_force_;
        }else{
            effective_force = Eigen::Vector3d::Zero();
        }

        double k_orient;
        double setStiffness;
        if(error.norm() > 0.03 && !reachedStart){
            setStiffness = 400.0;
            d_ = 2.0 * zeta * std::sqrt(m_ * setStiffness);
            k_orient   = 3.0;
            effective_force = Eigen::Vector3d::Zero();

            RCLCPP_INFO(this->get_logger(), "Error: %.4f", error.norm());
        }else{

            setStiffness = k_;
            d_ = 2.0 * zeta * std::sqrt(m_ * setStiffness);
            k_orient = 10.0;
            reachedStart = true;
        }
        // state.external_force.head<3>() is sitting right here whenever
        // you're ready to feed measured contact force into this admittance law.

        Eigen::Vector3d acceleration = (effective_force - (d_ * ee_vel_) - (setStiffness * error)) / m_;
        ee_vel_ += acceleration * dt;

        Eigen::Quaterniond q_err = target_orientation * state.ee_orientation.inverse();
        q_err.normalize();
        if (q_err.w() < 0.0) q_err.coeffs() = -q_err.coeffs();

        Eigen::Vector3d angular_vel = 2.0 * k_orient * q_err.vec();

        Eigen::VectorXd cart_vel(6);
        cart_vel << ee_vel_.x(), ee_vel_.y(), ee_vel_.z(),
                    angular_vel.x(), angular_vel.y(), angular_vel.z();

        double lambda = 0.05;
        Eigen::MatrixXd JJT = state.jacobian * state.jacobian.transpose();
        Eigen::MatrixXd J_pinv = state.jacobian.transpose() *
            (JJT + lambda * lambda * Eigen::MatrixXd::Identity(6,6)).inverse();

        Eigen::VectorXd q_target = J_pinv * cart_vel;
        q_dot = q_target - state.qdot;

        robotForceToHapticForce(state.external_force.head<3>());
        publishTargetVelocities(q_dot);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HapticControllerNode>());
    rclcpp::shutdown();
    return EXIT_SUCCESS;
}