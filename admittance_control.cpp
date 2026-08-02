/**
 * @file admittance_control_node.cpp
 * @brief Cartesian admittance controller for the Franka FR3 robot arm.
 *
 * This node implements an admittance control law in Cartesian space.
 * The controller receives a desired end-effector goal position and computes
 * joint velocities to move the robot compliantly toward that goal. External
 * forces (real or simulated) are fed into the admittance law to allow
 * compliant, human-friendly motion.
 *
 * Control Architecture:
 *   - Outer loop: Admittance law maps F_ext + position error → desired EE acceleration
 *   - Inner loop: Jacobian pseudoinverse maps desired EE velocity → joint velocities
 *   - Low-level:  Joint velocities sent to ign_ros2_control velocity controller
 *
 * Key Features:
 *   - Configurable virtual mass, damping, and stiffness via ROS 2 parameters
 *   - Optional critical damping auto-calculation: D = 2 * zeta * sqrt(M * K)
 *   - Virtual contact plane at z = 0.05 m (simulates table surface)
 *   - Gravity-compensated force reading from joint wrench topic
 *   - Damped least-squares Jacobian pseudoinverse for singularity robustness
 *   - Orientation stabilization (gripper pointing down)
 *   - CAN IGNORE : Code related to haptic device, all features related to the haptic device are DISABLED in this code. That was resused code i wrote from the Final Project, and was used to help with debugging.
 * 
 * ROS 2 Topics:
 *   Subscribed:
 *     - /goal_pose                   (geometry_msgs/Pose)          : Desired EE goal
 *     - /set_force                   (geometry_msgs/Vector3)        : Injected external force
 *     - /target_pose                 (geometry_msgs/PoseStamped)    : Topic input target (use_topic_input:=true)
 *     - /robot_force                 (geometry_msgs/Vector3)        : Topic input virtual force (use_topic_input:=true)
 *     - /joint_wrenches/fr3_joint7   (geometry_msgs/WrenchStamped)  : Raw EE wrench
 *     - /joint_states                (sensor_msgs/JointState)        : Joint feedback
 *   Published:
 *     - /target_joint_velocities     (std_msgs/Float64MultiArray)   : Joint velocity commands
 *     - ee_wrench                    (geometry_msgs/Vector3)         : Effective force in loop
 *     - ee_pos_error                 (geometry_msgs/Vector3)         : Cartesian position error
 *     - /ik_scaled_pose              (geometry_msgs/PoseStamped)     : Target EE pose (viz)
 *     - /robot/external_wrench       (geometry_msgs/WrenchStamped)   : Measured EE wrench (gui/bag)
 *     - /robot/ee_pose               (geometry_msgs/PoseStamped)     : Actual EE pose (gui/bag)
 *
 * Dependencies: ROS 2 Humble, MoveIt 2, Eigen3, franka_ros2, franka_gazebo
 *
 * @author Steven Yang
 * @course ME780 - Collaborative Robotics, Winter 2026
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <moveit_msgs/srv/get_position_fk.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int32.hpp>
#include <Eigen/Dense>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>

#include <rclcpp_action/rclcpp_action.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/move.hpp>
#include "franka_msgs/msg/franka_robot_state.hpp"

#include <fmt/format.h>

#include <deque>

// Global TF2 buffer and listener (used for gravity compensation frame transforms)
std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

class SimpleIKNode : public rclcpp::Node
{
public:
    SimpleIKNode() : Node("admittance_control_node")
    {
        // ─── 1. ROS 2 Parameters ──────────────────────────────────────────────
        // All admittance gains can be set at launch time or via a YAML config.
        this->declare_parameter<std::string>("base_frame", "panda_link0");
        this->declare_parameter<double>("position_scale_x", 4.0);   // haptic workspace scaling
        this->declare_parameter<double>("position_scale_y", 4.0);
        this->declare_parameter<double>("position_scale_z", 4.0);
        this->declare_parameter<double>("height_offset", 0.3);       // z offset for haptic mapping
        this->declare_parameter<double>("mass", 10.0);               // virtual inertia M_d (kg)
        this->declare_parameter<double>("damping", 140.0);           // virtual damping D_d (Ns/m)
        this->declare_parameter<double>("stiffness", 2000.0);        // virtual stiffness K_d (N/m)
        this->declare_parameter<double>("max_force_output", 5.0);    // haptic force clamp (N)
        this->declare_parameter<double>("valid_distance_threshold", 0.55);
        this->declare_parameter<bool>("use_gazebo", false);
        this->declare_parameter<bool>("use_topic_input", false);

        base_frame_               = this->get_parameter("base_frame").as_string();
        position_scale_x_         = this->get_parameter("position_scale_x").as_double();
        position_scale_y_         = this->get_parameter("position_scale_y").as_double();
        position_scale_z_         = this->get_parameter("position_scale_z").as_double();
        height_offset_            = this->get_parameter("height_offset").as_double();
        m_                        = this->get_parameter("mass").as_double();
        d_                        = this->get_parameter("damping").as_double();
        k_                        = this->get_parameter("stiffness").as_double();
        max_force_output_         = this->get_parameter("max_force_output").as_double();
        valid_distance_threshold_ = this->get_parameter("valid_distance_threshold").as_double();
        use_topic_input_          = this->get_parameter("use_topic_input").as_bool();
        if (use_topic_input_) {
            RCLCPP_INFO(this->get_logger(),
                "Topic input enabled: target from /target_pose, force from /robot_force");
        }

        // ─── 2. Critical Damping Auto-Calculation ────────────────────────────
        // Formula: D = 2 * zeta * sqrt(M * K)
        //   zeta < 1  → underdamped (oscillatory response)
        //   zeta = 1  → critically damped (fastest settling, no overshoot)
        //   zeta > 1  → overdamped (slow, no overshoot)
        // d_calculated is used in the admittance loop instead of d_ to ensure
        // the damping ratio is consistent with the chosen M and K.
        zeta         = 1.5;
        d_calculated = 2.0 * zeta * std::sqrt(m_ * k_);
        target_gripper_width_ = 0.04;

        mode_ = 0;

        // ─── 3. State Initialization ──────────────────────────────────────────
        ee_pos_      = Eigen::Vector3d(0.4, 0.0, 0.4);  // safe default start
        ee_vel_      = Eigen::Vector3d::Zero();
        haptic_goal_ = ee_pos_;
        set_goal_    = ee_pos_;
        robot_force_ = Eigen::Vector3d::Zero();
        measured_force_ = Eigen::Vector3d::Zero();
        set_force_   = Eigen::Vector3d::Zero();
        damping_force_ = Eigen::Vector3d::Zero();

        // ─── 4. Service Clients ───────────────────────────────────────────────
        ik_client_ = this->create_client<moveit_msgs::srv::GetPositionIK>("/compute_ik");
        fk_client_ = this->create_client<moveit_msgs::srv::GetPositionFK>("/compute_fk");

        // ─── 5. Subscriptions ─────────────────────────────────────────────────
        haptic_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "haptic/pose", 1,
            std::bind(&SimpleIKNode::hapticPoseCallback, this, std::placeholders::_1));

        goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "/set_goal_pose", 10,
            std::bind(&SimpleIKNode::setGoalPoseCallback, this, std::placeholders::_1));

        set_force_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/set_force", 10,
            std::bind(&SimpleIKNode::setForceCallback, this, std::placeholders::_1));

        //joint_wrench_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>("/joint_wrenches/fr3_joint7", 10,std::bind(&SimpleIKNode::jointWrenchCallback, this, std::placeholders::_1));

        franka_state_sub_ = this->create_subscription<franka_msgs::msg::FrankaRobotState>(
        "/franka_robot_state_broadcaster/robot_state", 1,
        std::bind(&SimpleIKNode::frankaRobotStateCallback, this, std::placeholders::_1));

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 1,
            std::bind(&SimpleIKNode::jointStateCallback, this, std::placeholders::_1));

        button_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/haptic/buttons", 10,
            std::bind(&SimpleIKNode::buttonCallback, this, std::placeholders::_1));

        gripper_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            "/target_gripper_width", 10);

        mode_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/toggle_mode", 10,
            [this](const std_msgs::msg::Int32::SharedPtr msg) {
                mode_ = (mode_ == 0) ? 1 : 0;
                RCLCPP_INFO(this->get_logger(), "Mode toggled → %d (%s)",
                    mode_, mode_ == 0 ? "HOME pose" : "Haptic follow");
            });

        stiffness_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "/stiffness", 10,
            std::bind(&SimpleIKNode::stiffnessCallback, this, std::placeholders::_1));

        // ── Topic-driven input (use_topic_input:=true) ────────────────────────
        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target_pose", 1,
            std::bind(&SimpleIKNode::targetPoseCallback, this, std::placeholders::_1));

        robot_force_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/robot_force", 1,
            std::bind(&SimpleIKNode::robotForceCallback, this, std::placeholders::_1));

        // ─── 6. Publishers ────────────────────────────────────────────────────
        haptic_force_pub_        = this->create_publisher<geometry_msgs::msg::Vector3>("/haptic/force_command", 1);
        //target_joint_pub_        = this->create_publisher<std_msgs::msg::Float64MultiArray>("/target_joint_positions", 10);
        target_joint_vel_pub_    = this->create_publisher<std_msgs::msg::Float64MultiArray>("/target_joint_velocities", 1);
        //target_joint_torque_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/target_joint_torques", 10);
        scaled_pose_pub_         = this->create_publisher<geometry_msgs::msg::PoseStamped>("/ik_scaled_pose", 10);
        wrench_pub_              = this->create_publisher<geometry_msgs::msg::Vector3>("ee_wrench", 10);
        force_mag_pub_          = this->create_publisher<std_msgs::msg::Float64>("ee_force_mag", 10);
        ee_pos_error_pub_        = this->create_publisher<geometry_msgs::msg::Vector3>("ee_pos_error", 10);
        ext_wrench_pub_          = this->create_publisher<geometry_msgs::msg::WrenchStamped>("/robot/external_wrench", 10);
        ee_pose_pub_             = this->create_publisher<geometry_msgs::msg::PoseStamped>("/robot/ee_pose", 10);

        // ─── 7. TF2 Setup ─────────────────────────────────────────────────────
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // ─── 8. Control Timer (1000 Hz) ─────────────────────────────────────────
        admittance_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1),
            std::bind(&SimpleIKNode::admittanceLoop, this));

        last_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "Admittance Controller Running.");

        // Safe default pose for IK seed state
        last_valid_pose_.header.frame_id = base_frame_;
        last_valid_pose_.pose.position.x = 0.3;
        last_valid_pose_.pose.position.y = 0.0;
        last_valid_pose_.pose.position.z = 0.3;
        last_valid_pose_.pose.orientation.w = 1.0;
    }

    /**
     * @brief Loads the MoveIt robot model and initializes the kinematic state.
     * Must be called after node construction (requires shared_from_this()).
     */
    void initializeRobotModel() {
        robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(
            shared_from_this(), "robot_description");
        kinematic_model_ = robot_model_loader_->getModel();
        kinematic_state_ = std::make_shared<moveit::core::RobotState>(kinematic_model_);
        kinematic_state_->setToDefaultValues();
        joint_model_group_ = kinematic_model_->getJointModelGroup("panda_arm");
        RCLCPP_INFO(this->get_logger(), "Robot model loaded for Jacobian computation.");

        const std::vector<std::string>& joint_names = joint_model_group_->getActiveJointModelNames();

        for (const std::string& joint_name : joint_names) {
            // Get the specific joint model
            const moveit::core::JointModel* joint_model = kinematic_model_->getJointModel(joint_name);
            
            if (joint_model) {
                // Get the bounds vector (stores min/max for each variable of the joint)
                const moveit::core::JointModel::Bounds& bounds = joint_model->getVariableBounds();
                
                // Single DOF joints (like Panda rotational joints) have exactly 1 bound entry at index 0
                if (!bounds.empty()) {
                    double min_limit = bounds[0].min_position_;
                    double max_limit = bounds[0].max_position_;
                    
                    RCLCPP_INFO(this->get_logger(), "Joint '%s' limits: Min = %f, Max = %f rad", 
                                joint_name.c_str(), min_limit, max_limit);
                }
            }
        }

    }

private:

    // ─────────────────────────────────────────────────────────────────────────
    // CALLBACKS
    // ─────────────────────────────────────────────────────────────────────────

    void stiffnessCallback(const std_msgs::msg::Float64::SharedPtr msg) {
        k_ = msg->data;
        d_calculated = 2.0 * zeta * std::sqrt(m_ * k_);
        RCLCPP_INFO(this->get_logger(), "Stiffness updated: K = %.2f, D (calculated) = %.2f", k_, d_calculated);
    }
    /**
     * @brief Maps haptic device pose to robot goal with axis remapping and scaling.
     * The haptic device coordinate frame differs from the robot base frame,
     * so axes are remapped and scaled to match the robot workspace.
     */
    void hapticPoseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
        if (use_topic_input_) {
            return;
        }
        latest_haptic_pose_ = *msg;
        haptic_goal_.x() = (-msg->position.z * position_scale_x_) + 0.35; //0.3
        haptic_goal_.y() = -msg->position.x * position_scale_y_;
        haptic_goal_.z() = (msg->position.y * position_scale_z_) + height_offset_;

        // Workspace boundaries (robot frame)
        const double x_min=0.2, x_max=0.7;
        const double y_min=-0.25, y_max=0.25;
        const double z_min=0.03, z_max=0.70;

        // ── Boundary force feedback ───────────────────────────────────────────
        // Compute penetration depth past each boundary, output restoring force
        // proportional to penetration. Force is in robot frame, then remapped
        // to haptic device frame (same axis mapping as robotForceToHapticForce).
        const double k_wall = 200.0;  // wall stiffness (N/m) — tune to feel
        Eigen::Vector3d boundary_force = Eigen::Vector3d::Zero();

        // X axis
        if      (haptic_goal_.x() < x_min) boundary_force.x() =  k_wall * (x_min - haptic_goal_.x());
        else if (haptic_goal_.x() > x_max) boundary_force.x() = -k_wall * (haptic_goal_.x() - x_max);

        // Y axis
        if      (haptic_goal_.y() < y_min) boundary_force.y() =  k_wall * (y_min - haptic_goal_.y());
        else if (haptic_goal_.y() > y_max) boundary_force.y() = -k_wall * (haptic_goal_.y() - y_max);

        // Z axis
        if      (haptic_goal_.z() < z_min) boundary_force.z() =  k_wall * (z_min - haptic_goal_.z());
        else if (haptic_goal_.z() > z_max) boundary_force.z() = -k_wall * (haptic_goal_.z() - z_max);

        // Remap robot frame → haptic device frame and publish
        if (boundary_force.norm() > 0.0) {
            geometry_msgs::msg::Vector3 force_out;
            force_out.x = -boundary_force.y();
            force_out.y =  boundary_force.z();
            force_out.z = -boundary_force.x();

            // Clamp magnitude
            double mag = std::sqrt(force_out.x*force_out.x +
                                force_out.y*force_out.y +
                                force_out.z*force_out.z);
            if (mag > max_force_output_) {
                double scale = max_force_output_ / mag;
                force_out.x *= scale;
                force_out.y *= scale;
                force_out.z *= scale;
            }

            //haptic_force_pub_->publish(force_out);
        }

        // Hard clamp after force is computed (so penetration depth is accurate)
        haptic_goal_.x() = std::clamp(haptic_goal_.x(), x_min, x_max);
        haptic_goal_.y() = std::clamp(haptic_goal_.y(), y_min, y_max);
        haptic_goal_.z() = std::clamp(haptic_goal_.z(), z_min, z_max);

        // Convert haptic orientation from message to Eigen
        Eigen::Quaterniond haptic_orientation(
            latest_haptic_pose_.orientation.w,
            latest_haptic_pose_.orientation.x,
            latest_haptic_pose_.orientation.y,
            latest_haptic_pose_.orientation.z
        );

        Eigen::Quaterniond rot_1(Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond rot_0(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond base_down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
        haptic_orientation_goal_ = (rot_1 * haptic_orientation * rot_0 * base_down).normalized();

        //applyHapticDamping(*msg);
    }
    /**
     * @brief Receives a desired Cartesian goal pose published to /set_goal_pose.
     * Used in simulation experiments to command the robot to target positions.
     */
    void setGoalPoseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "GOAL SET MANUALLY");

        set_goal_.x() = msg->position.x;
        set_goal_.y() = msg->position.y;
        set_goal_.z() = msg->position.z;
    }

    /**
     * @brief Manually injects an external force vector into the admittance law.
     * Used to simulate human disturbances (e.g. a push) without a physical sensor.
     */
    void setForceCallback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "FORCE SET MANUALLY");
        set_force_.x() = msg->x;
        set_force_.y() = msg->y;
        set_force_.z() = msg->z;
    }

    /**
     * @brief Commands the controller goal via the /target_pose topic.
     * Active only when use_topic_input:=true; ignored otherwise.
     */
    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        if (!use_topic_input_) {
            return;
        }
        haptic_goal_.x() = msg->pose.position.x;
        haptic_goal_.y() = msg->pose.position.y;
        haptic_goal_.z() = msg->pose.position.z;
        haptic_orientation_goal_ = Eigen::Quaterniond(
            msg->pose.orientation.w, msg->pose.orientation.x,
            msg->pose.orientation.y, msg->pose.orientation.z);
        RCLCPP_INFO(this->get_logger(), "Target pose set: [%.3f, %.3f, %.3f]",
                    haptic_goal_.x(), haptic_goal_.y(), haptic_goal_.z());
    }

    /**
     * @brief Commands the virtual external force via the /robot_force topic.
     * Active only when use_topic_input:=true; ignored otherwise.
     */
    void robotForceCallback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
        if (!use_topic_input_) {
            return;
        }
        robot_force_.x() = msg->x;
        robot_force_.y() = msg->y;
        robot_force_.z() = msg->z;
        RCLCPP_INFO(this->get_logger(), "Robot force set: [%.3f, %.3f, %.3f]",
                    robot_force_.x(), robot_force_.y(), robot_force_.z());
    }

    /**
     * @brief Processes raw joint wrench and computes gravity-compensated EE force.
     *
     * Pipeline:
     *   1. Read raw force from joint torque sensor at fr3_joint7
     *   2. Rotate gravity vector into sensor frame using TF2 transform
     *   3. Subtract gravitational force contribution (weight of EE + payload)
     *   4. Subtract velocity-proportional damping to reduce motion-induced noise
     *
     * IMPORTANT: Force sensor quality directly impacts admittance controller stability.
     * Noisy, delayed, or erroneous measurements are treated as legitimate interaction
     * forces by the admittance law. Gravity compensation and damping correction are
     * applied here to improve signal quality before use in the control loop.
     */
    void jointWrenchCallback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
        Eigen::Vector3d raw_force(
            msg->wrench.force.x,
            msg->wrench.force.y,
            msg->wrench.force.z
        );

        // Get rotation matrix from base frame to sensor frame
        Eigen::Matrix3d R_base_to_sensor = getRotationToBase();

        // Gravity in world frame (z-down)
        Eigen::Vector3d gravity_vec(0, 0, -9.81);

        // Project gravity into sensor frame and compute weight contribution
        Eigen::Vector3d weight_in_sensor =
            R_base_to_sensor.transpose() * (total_mass * gravity_vec);

        // Velocity-dependent correction to reduce motion-induced measurement artifacts
        Eigen::Vector3d ee_vel_sensor = R_base_to_sensor.transpose() * ee_vel_;
        const double D = 5.0;
        Eigen::Vector3d damping_force = D * ee_vel_sensor;

        // Final corrected force: remove gravity and motion artifacts
        robot_force_ = raw_force + weight_in_sensor - damping_force;

        geometry_msgs::msg::Vector3 forceMsg;
        forceMsg.x = robot_force_.x();
        forceMsg.y = robot_force_.y();
        forceMsg.z = robot_force_.z();
    }

    void frankaRobotStateCallback(const franka_msgs::msg::FrankaRobotState::SharedPtr msg) {
        // o_f_ext_hat_k is a std::array<double, 6>: [fx, fy, fz, tx, ty, tz]
        Eigen::Vector3d raw_force(
            -msg->o_f_ext_hat_k[0],  // fx
            -msg->o_f_ext_hat_k[1],  // fy
            -msg->o_f_ext_hat_k[2]   // fz
        );

        // Get rotation matrix from base frame to sensor frame
        Eigen::Matrix3d R_base_to_sensor = getRotationToBase();

        // Gravity in world frame (z-down)
        Eigen::Vector3d gravity_vec(0, 0, -9.81);

        // Project gravity into sensor frame and compute weight contribution
        Eigen::Vector3d weight_in_sensor =
            R_base_to_sensor.transpose() * (2.35/9.81)*gravity_vec; //(total_mass * gravity_vec)

        // Velocity-dependent correction to reduce motion-induced measurement artifacts
        Eigen::Vector3d ee_vel_sensor = R_base_to_sensor.transpose() * ee_vel_;
        const double B = 0.0;
        Eigen::Vector3d damping_force = B * ee_vel_;//ee_vel_sensor;

        // Final corrected force: remove gravity and motion artifacts
        measured_force_ = raw_force;
        if (!use_topic_input_) {
            robot_force_ = raw_force;
        }
        Eigen::Vector3d effective_force = Eigen::Vector3d::Zero();
        effective_force = 0.25*robot_force_; //- damping_force;//-damping_force_; //0.5*robot_force_ + damping_force_;
        /*
        geometry_msgs::msg::Vector3 forceMsg;
        forceMsg.x = robot_force_.x();
        forceMsg.y = robot_force_.y();
        forceMsg.z = robot_force_.z();

        wrench_pub_->publish(forceMsg);  // moved inside callback
        */
        //haptic_force_pub_->publish(effective_force);
        geometry_msgs::msg::Vector3 forceMsg;
        forceMsg.x = robot_force_.x(); //robot_force_
        forceMsg.y = robot_force_.y();
        forceMsg.z = robot_force_.z();
        wrench_pub_->publish(forceMsg);  // moved inside callback

        std_msgs::msg::Float64 mag_msg;
        mag_msg.data = robot_force_.norm();
        force_mag_pub_->publish(mag_msg);

        // Publish measured wrench on the shared interface for gui_controller /
        // bag recording. Uses the raw measurement so the recorded force reflects
        // the true sensor reading even when a /robot_force command is injected.
        geometry_msgs::msg::WrenchStamped wrench_msg;
        wrench_msg.header.stamp    = this->now();
        wrench_msg.header.frame_id = base_frame_;
        wrench_msg.wrench.force.x  = measured_force_.x();
        wrench_msg.wrench.force.y  = measured_force_.y();
        wrench_msg.wrench.force.z  = measured_force_.z();
        wrench_msg.wrench.torque.x = msg->o_f_ext_hat_k[3];
        wrench_msg.wrench.torque.y = msg->o_f_ext_hat_k[4];
        wrench_msg.wrench.torque.z = msg->o_f_ext_hat_k[5];
        ext_wrench_pub_->publish(wrench_msg);

        if (!use_topic_input_) robotForceToHapticForce(effective_force);
    }

    /**
     * @brief Converts robot-frame force to haptic device frame and publishes it.
     *
     * Applies axis remapping, optional IIR low-pass filter, optional moving
     * average filter, magnitude scaling, and output clamping.
     */
    void robotForceToHapticForce(Eigen::Vector3d robot_force_in)
    {
        // Axis remapping: robot frame → haptic device frame
        geometry_msgs::msg::Vector3 force_out;
        force_out.x = -robot_force_in.y();
        force_out.y =  robot_force_in.z();
        force_out.z = -robot_force_in.x();

        const bool USE_LOW_PASS_FILTER  = true;
        const bool USE_MOVING_AVERAGE   = false;
        const bool FORCE_OUTPUT_ENABLED = true;

        // ── IIR Low-pass filter: y[n] = alpha*x[n] + (1-alpha)*y[n-1] ────────
        // Lower alpha = more smoothing but more lag
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

        // ── Moving average filter (alternative to low-pass) ───────────────────
        const size_t MA_WINDOW_SIZE = 5;
        static std::deque<geometry_msgs::msg::Vector3> ma_window;
        if (USE_MOVING_AVERAGE) {
            ma_window.push_back(force_out);
            if (ma_window.size() > MA_WINDOW_SIZE) ma_window.pop_front();
            if (ma_window.size() == MA_WINDOW_SIZE) {
                double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
                for (const auto& v : ma_window) { sum_x += v.x; sum_y += v.y; sum_z += v.z; }
                force_out.x = sum_x / MA_WINDOW_SIZE;
                force_out.y = sum_y / MA_WINDOW_SIZE;
                force_out.z = sum_z / MA_WINDOW_SIZE;
            }
        }

        // ── Scale down to limit noise influence ───────────────────────────────
        //force_out.x *= 0.2;
        //force_out.y *= 0.2;
        //force_out.z *= 0.2;

        // ── Clamp magnitude while preserving direction ────────────────────────
        double magnitude = std::sqrt(
            force_out.x * force_out.x +
            force_out.y * force_out.y +
            force_out.z * force_out.z);
        if (magnitude > max_force_output_) {
            double scale = max_force_output_ / magnitude;
            force_out.x *= scale; force_out.y *= scale; force_out.z *= scale;
        }

        if (!FORCE_OUTPUT_ENABLED) force_out.x = force_out.y = force_out.z = 0.0;

        /*
        geometry_msgs::msg::Vector3 force_damped;
        force_damped.x = damping_force_.x() + force_out.x;
        force_damped.y = damping_force_.y() + force_out.y;
        force_damped.z = damping_force_.z() + force_out.z;
        */

        //force_damped += force_out;//force_out 

        haptic_force_pub_->publish(force_out); //force_damped
    }

    void applyHapticDamping(const geometry_msgs::msg::Pose& current_pose)
    {
        const bool DAMPING_ENABLED = true;

        // ── Velocity estimation via finite difference ─────────────────────────
        static geometry_msgs::msg::Pose prev_pose;
        static rclcpp::Time prev_time;
        static bool initialized = false;

        rclcpp::Time now = this->now();

        if (!initialized) {
            prev_pose = current_pose;
            prev_time = now;
            initialized = true;
            return;
        }

        double dt = (now - prev_time).seconds();
        if (dt <= 0.0 || dt > 0.1) {
            prev_pose = current_pose;
            prev_time = now;
            return;
        }

        // Raw velocity in haptic device frame
        Eigen::Vector3d haptic_vel(
            (current_pose.position.x - prev_pose.position.x) / dt,
            (current_pose.position.y - prev_pose.position.y) / dt,
            (current_pose.position.z - prev_pose.position.z) / dt
        );

        prev_pose = current_pose;
        prev_time = now;

        // ── IIR low-pass filter on velocity (reduces differentiation noise) ───
        static Eigen::Vector3d filtered_vel = Eigen::Vector3d::Zero();
        const double alpha = 0.15;  // lower = smoother but more lag
        filtered_vel = alpha * haptic_vel + (1.0 - alpha) * filtered_vel;

        // ── Damping force: F = -B * v ─────────────────────────────────────────
        const double B = 15.0;  // damping coefficient (N·s/m) — tune to feel
        damping_force_ = -B * filtered_vel;

        // ── Clamp magnitude while preserving direction ────────────────────────
        double magnitude = damping_force_.norm();
        if (magnitude > max_force_output_) {
            damping_force_ *= (max_force_output_ / magnitude);
        }

        // ── Publish ───────────────────────────────────────────────────────────
        geometry_msgs::msg::Vector3 force_out;
        force_out.x = DAMPING_ENABLED ? damping_force_.x() : 0.0;
        force_out.y = DAMPING_ENABLED ? damping_force_.y() : 0.0;
        force_out.z = DAMPING_ENABLED ? damping_force_.z() : 0.0;

        //haptic_force_pub_->publish(force_out);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MAIN ADMITTANCE CONTROL LOOP (50 Hz)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * @brief Primary admittance control loop, called at 50 Hz.
     *
     * Steps:
     *   1.  Timestep guard (reject invalid dt)
     *   2.  Update MoveIt kinematic state from latest joint positions
     *   3.  Forward kinematics to get actual EE position
     *   4.  Compute position error: e = x_actual - x_goal
     *   5.  Determine effective force (sensor / manual / virtual contact)
     *   6.  Admittance law: a = (F_ext - D*v - K*e) / M
     *   7.  Euler integration: v += a*dt,  x += v*dt
     *   8.  Orientation control (quaternion error → angular velocity)
     *   9.  Damped least-squares Jacobian pseudoinverse
     *   10. Safety clamp on joint velocities
     *   11. Publish joint velocity command
     */
    void admittanceLoop() {
        // ── Timestep guard ────────────────────────────────────────────────────
        auto now = this->now();
        double dt = (now - last_time_).seconds();
        last_time_ = now;
        if (dt <= 0.0 || dt > 0.1) return;
        if (!kinematic_state_ || !joint_model_group_) return;

        // ── 1. Sync MoveIt state with current joint positions ─────────────────
        if (latest_joint_state_.name.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "No joint state received yet, skipping");
            return;
        }

        std::vector<std::string> arm_joints = {
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        };
        for (const auto& name : arm_joints) {
            auto it = std::find(latest_joint_state_.name.begin(),
                                latest_joint_state_.name.end(), name);
            if (it != latest_joint_state_.name.end()) {
                size_t idx = std::distance(latest_joint_state_.name.begin(), it);
                kinematic_state_->setJointPositions(name, &latest_joint_state_.position[idx]);
            }
        }
        kinematic_state_->updateLinkTransforms();

        // ── 2. Forward kinematics: actual EE position ─────────────────────────
        const Eigen::Isometry3d& ee_transform =
            kinematic_state_->getGlobalLinkTransform("panda_hand_tcp");
        actual_ee_pos_ = ee_transform.translation();
        actual_ee_pos_initialized_ = true;

        if (mode_ == 0) {

            reachedStart = false;
            // ── Mode 0: joint-space P control to home pose ────────────────────
            const std::vector<double> home_joints =
                {0.0, 0.0, 0.0, -M_PI/2.0, 0.0, M_PI/2.0, M_PI/4.0};
            const std::vector<std::string> arm_joints_ordered = {
                "panda_joint1","panda_joint2","panda_joint3","panda_joint4",
                "panda_joint5","panda_joint6","panda_joint7"};

            const double k_home = 1.0;   // proportional gain (rad/s per rad error)
            const double max_home_vel = 0.5;

            Eigen::VectorXd q_dot(7);
            for (int i = 0; i < 7; i++) {
                auto it = std::find(latest_joint_state_.name.begin(),
                                    latest_joint_state_.name.end(),
                                    arm_joints_ordered[i]);
                double q_current = 0.0;
                if (it != latest_joint_state_.name.end())
                    q_current = latest_joint_state_.position[
                        std::distance(latest_joint_state_.name.begin(), it)];

                double err = home_joints[i] - q_current;
                q_dot[i] = std::clamp(k_home * err, -max_home_vel, max_home_vel);
            }

            // Reset admittance integrator to actual EE pos so mode-1
            // re-entry doesn't jump
            ee_pos_  = actual_ee_pos_;
            ee_vel_  = Eigen::Vector3d::Zero();
            haptic_goal_ = actual_ee_pos_;

            publishTargetJointVelocities(q_dot);
            return;   // skip the rest of the Cartesian admittance path
        }

        // ── 3. Position error: e = x_actual - x_goal ─────────────────────────
        Eigen::Vector3d error = actual_ee_pos_ - haptic_goal_;

        // ── 4. Effective force determination ─────────────────────────────────
        Eigen::Vector3d effective_force = Eigen::Vector3d::Zero();
        bool readForce = true; //read for from FT sensor

        if (readForce) {
            // Use gravity-compensated force sensor reading (scaled conservatively)
            effective_force = 0.5 * robot_force_;
        } else {
            // Blend small sensor contribution with manually set force
            effective_force = 0.0 * robot_force_;

            /*
            effective_force = 0.01 * robot_force_ + set_force_;

            // ── Virtual contact plane at z = 0.05 m ──────────────────────────
            // Simulates a rigid table surface below the robot. When the EE
            // penetrates below plane_z, a spring force pushes it back upward.
            // F_contact = table_k * penetration_depth (normal direction only)
            const double plane_z     = 0.05;    // table surface height (m)
            const double table_k     = 5000.0;  // contact stiffness (N/m)
            const double penetration = plane_z - actual_ee_pos_.z();
            if (penetration > 0.0) {
                effective_force.z() += table_k * penetration;
                //RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 200,"Virtual contact: penetration=%.4f m, Fz=%.2f N",penetration, table_k * penetration);
            }
            */
            
        }

        // ── 5. Publish effective force and position error for data logging ────
        geometry_msgs::msg::Vector3 forceMsg;
        forceMsg.x = effective_force.x();
        forceMsg.y = effective_force.y();
        forceMsg.z = effective_force.z();
        //wrench_pub_->publish(forceMsg);

        geometry_msgs::msg::Vector3 posErrorMsg;
        posErrorMsg.x = error.x();
        posErrorMsg.y = error.y();
        posErrorMsg.z = error.z();
        ee_pos_error_pub_->publish(posErrorMsg);

        //RCLCPP_INFO(this->get_logger(), "Error length: %f", error.norm());

        double k_orient;
        double setStiffness;
        if(error.norm() > 0.03 && !reachedStart){
            setStiffness = 300.0;
            k_orient   = 3.0;
            effective_force = Eigen::Vector3d::Zero();
        }else{

            setStiffness = k_;
            k_orient = 10.0;
            reachedStart = true;
        }
        // ── 6. Admittance law: a = (F_ext - D*v - K*e) / M ──────────────────
        // d_calculated ensures the damping ratio zeta is maintained regardless
        // of the chosen M and K values.
        Eigen::Vector3d acceleration =
            (effective_force - (d_calculated * ee_vel_) - (setStiffness * error)) / m_;

        // ── 7. Euler integration ──────────────────────────────────────────────
        ee_vel_ += acceleration * dt;
        ee_pos_ += ee_vel_ * dt;

        // ── 8. Target orientation: gripper pointing straight down or haptic device
        Eigen::Quaterniond target_orientation;

        target_orientation = haptic_orientation_goal_;

        // Publish target pose for RViz visualization
        geometry_msgs::msg::PoseStamped target_pose;
        target_pose.header.stamp    = now;
        target_pose.header.frame_id = base_frame_;
        target_pose.pose.position.x = ee_pos_.x();
        target_pose.pose.position.y = ee_pos_.y();
        target_pose.pose.position.z = ee_pos_.z();
        target_pose.pose.orientation.x = target_orientation.x();
        target_pose.pose.orientation.y = target_orientation.y();
        target_pose.pose.orientation.z = target_orientation.z();
        target_pose.pose.orientation.w = target_orientation.w();
        scaled_pose_pub_->publish(target_pose);

        // ── 9. Compute Jacobian J (6x7) at current configuration ─────────────
        Eigen::MatrixXd J;
        kinematic_state_->getJacobian(
            joint_model_group_,
            kinematic_state_->getLinkModel("panda_hand_tcp"),
            Eigen::Vector3d::Zero(), J);

        // ── 10. Orientation error → angular velocity command ─────────────────
        Eigen::Quaterniond current_orientation(ee_transform.rotation());
        Eigen::Quaterniond q_err = target_orientation * current_orientation.inverse();
        q_err.normalize();
        if (q_err.w() < 0.0) q_err.coeffs() = -q_err.coeffs();  // shortest-path convention

        // Publish actual EE pose on the shared interface for gui_controller /
        // bag recording.
        geometry_msgs::msg::PoseStamped ee_pose_msg;
        ee_pose_msg.header.stamp    = now;
        ee_pose_msg.header.frame_id = base_frame_;
        ee_pose_msg.pose.position.x = actual_ee_pos_.x();
        ee_pose_msg.pose.position.y = actual_ee_pos_.y();
        ee_pose_msg.pose.position.z = actual_ee_pos_.z();
        ee_pose_msg.pose.orientation.x = current_orientation.x();
        ee_pose_msg.pose.orientation.y = current_orientation.y();
        ee_pose_msg.pose.orientation.z = current_orientation.z();
        ee_pose_msg.pose.orientation.w = current_orientation.w();
        ee_pose_pub_->publish(ee_pose_msg);

        const double max_ang_vel = 0.5;
        Eigen::Vector3d angular_vel = 2.0 * k_orient * q_err.vec();
        //angular_vel = angular_vel.cwiseMax(-max_ang_vel).cwiseMin(max_ang_vel);

        // ── 11. Build 6D Cartesian velocity [vx vy vz wx wy wz] ──────────────
        Eigen::VectorXd cart_vel(6);
        cart_vel << ee_vel_.x(), ee_vel_.y(), ee_vel_.z(),
                    angular_vel.x(), angular_vel.y(), angular_vel.z();

        // ── 12. Damped least-squares pseudoinverse: J† = J^T(JJ^T + λ²I)^-1 ─
        // λ regularizes the inversion near singularities, preventing large
        // joint velocity spikes when the Jacobian is rank-deficient.
        const double lambda = 0.05;
        Eigen::MatrixXd JJT   = J * J.transpose();
        Eigen::MatrixXd J_pinv = J.transpose() *
            (JJT + lambda * lambda * Eigen::MatrixXd::Identity(6, 6)).inverse();
        Eigen::VectorXd q_dot = J_pinv * cart_vel;

        // ── 13. Safety clamp joint velocities ────────────────────────────────
        const double max_joint_vel = 1.0;  // rad/s (conservative)
        for (int i = 0; i < 7; i++) q_dot[i] = std::clamp(q_dot[i], -max_joint_vel, max_joint_vel);

        //RCLCPP_INFO(this->get_logger(),"Joint 6 Value: [%.3f]",latest_joint_state_.position[5]);
        if(latest_joint_state_.position[6] < ((-150)*(M_PI/180)) || latest_joint_state_.position[6] > (150*(M_PI/180)) || latest_joint_state_.position[5] > (200*(M_PI/180)) || latest_joint_state_.position[3] > ((-40)*(M_PI/180)) || latest_joint_state_.position[3] < ((-170)*(M_PI/180))){
            mode_ = 0;
        }
        // ── 14. Publish to low-level velocity controller ──────────────────────
        publishTargetJointVelocities(q_dot);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HELPER METHODS
    // ─────────────────────────────────────────────────────────────────────────

    /** @brief Returns true if the proposed movement exceeds the safety threshold. */
    bool isMovementAllowed(const geometry_msgs::msg::Point& current,
                           const geometry_msgs::msg::Point& last_valid)
    {
        double dx = current.x - last_valid.x;
        double dy = current.y - last_valid.y;
        double dz = current.z - last_valid.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz) > valid_distance_threshold_;
    }

    /** @brief Publishes 7-DOF joint velocity command to the velocity controller. */
    void publishTargetJointVelocities(const Eigen::VectorXd& q_dot)
    {
        std_msgs::msg::Float64MultiArray msg;
        msg.data.resize(7);
        for (int i = 0; i < 7; i++) msg.data[i] = q_dot[i];
        target_joint_vel_pub_->publish(msg);
        //RCLCPP_INFO(this->get_logger(),"Joint velocities: [%.3f %.3f %.3f %.3f %.3f %.3f %.3f]",q_dot[0], q_dot[1], q_dot[2], q_dot[3], q_dot[4], q_dot[5], q_dot[6]);
    }

    /**
     * @brief Builds MoveIt RobotState from latest joint state.
     * Falls back to Franka home configuration if no joint state has been received.
     */
    moveit_msgs::msg::RobotState getCurrentRobotState()
    {
        moveit_msgs::msg::RobotState robot_state;
        robot_state.is_diff = true;
        if (!latest_joint_state_.name.empty()) {
            robot_state.joint_state = latest_joint_state_;
        } else {
            RCLCPP_WARN(this->get_logger(), "No joint state received, using defaults");
            sensor_msgs::msg::JointState js;
            js.name = {"panda_joint1","panda_joint2","panda_joint3","panda_joint4",
            "panda_joint5","panda_joint6","panda_joint7"};
            js.position = {0, -0.785, 0, -2.356, 0, 1.571, 0.785};  // Franka home
            robot_state.joint_state = js;
        }
        return robot_state;
    }

    /**
     * @brief Asynchronous IK request via MoveIt compute_ik service.
     * @param target_pose  Desired EE pose in base frame
     * @param callback     Called with joint solution or nullopt on failure
     */
    void computeIK(const geometry_msgs::msg::PoseStamped& target_pose,
                   std::function<void(std::optional<sensor_msgs::msg::JointState>)> callback)
    {
        auto req = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
        req->ik_request.group_name       = "panda_arm";
        req->ik_request.avoid_collisions = false;
        req->ik_request.pose_stamped     = target_pose;
        req->ik_request.timeout.sec      = 2;
        req->ik_request.robot_state      = getCurrentRobotState();

        ik_client_->async_send_request(req,
            [callback, this](rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedFuture f) {
                auto res = f.get();
                if (res->error_code.val == res->error_code.SUCCESS)
                    callback(res->solution.joint_state);
                else {
                    RCLCPP_WARN(this->get_logger(), "IK failed: %d", res->error_code.val);
                    callback(std::nullopt);
                }
            });
    }

    /**
     * @brief Retrieves rotation matrix from base frame to fr3_link7 via TF2.
     * Used in gravity compensation to project gravity into the sensor frame.
     * Falls back to identity matrix if the transform is unavailable.
     */
    Eigen::Matrix3d getRotationToBase()
    {
        try {
            auto tf = tf_buffer_->lookupTransform(base_frame_, "panda_link7", tf2::TimePointZero);
            const auto& q = tf.transform.rotation;
            tf2::Quaternion tq(q.x, q.y, q.z, q.w);
            tf2::Matrix3x3 tm(tq);
            Eigen::Matrix3d rot;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    rot(i,j) = tm[i][j];

            //RCLCPP_INFO(this->get_logger(), "TF lookup success");

            return rot;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
            return Eigen::Matrix3d::Identity();
        }
    }

    /**
     * @brief Asynchronously updates actual EE position via FK service.
     * Used in the torque-based control loop (admittanceLoop2).
     */
    void updateActualEEPosition() {
        if (!fk_client_->service_is_ready() || latest_joint_state_.name.empty()) return;
        auto req = std::make_shared<moveit_msgs::srv::GetPositionFK::Request>();
        req->header.frame_id = base_frame_;
        req->fk_link_names.push_back("panda_hand_tcp");
        req->robot_state.joint_state = latest_joint_state_;
        fk_client_->async_send_request(req,
            [this](rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedFuture f) {
                auto res = f.get();
                if (res->error_code.val == res->error_code.SUCCESS) {
                    const auto& p = res->pose_stamped[0].pose.position;
                    actual_ee_pos_ = {p.x, p.y, p.z};
                    actual_ee_pos_initialized_ = true;
                }
            });
    }

    /** @brief Publishes joint positions from IK solution (position control path). */
    void publishJoints(const sensor_msgs::msg::JointState& sol) {
        std_msgs::msg::Float64MultiArray msg;
        std::vector<std::string> names = {"panda_joint1","panda_joint2","panda_joint3",
                                  "panda_joint4","panda_joint5","panda_joint6","panda_joint7"};
        for (auto& n : names) {
            auto it = std::find(sol.name.begin(), sol.name.end(), n);
            if (it != sol.name.end())
                msg.data.push_back(sol.position[std::distance(sol.name.begin(), it)]);
        }
        target_joint_pub_->publish(msg);
    }

    /** @brief Stores latest joint state for use in the control loop. */
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        latest_joint_state_ = *msg;
    }

    void buttonCallback(const std_msgs::msg::Int32::SharedPtr msg)
    {
        /*
        
        */
        // Determine if we're in Gazebo simulation or real hardware
        bool is_gazebo = false;
        this->get_parameter("use_gazebo", is_gazebo);
        
        if (is_gazebo) {
            // Gazebo mode: Use simple topic-based gripper control
            std_msgs::msg::Float64 gripper_msg;
            
            if ((msg->data & 1) != 0) {
                gripper_msg.data = 0.0;  // Close gripper
                RCLCPP_DEBUG(this->get_logger(), "Button pressed - closing gripper (Gazebo)");
            } else {
                gripper_msg.data = 0.03; // Open gripper
                RCLCPP_DEBUG(this->get_logger(), "Button released - opening gripper (Gazebo)");
            }
            
            gripper_pub_->publish(gripper_msg);
        } else {
            // Real hardware mode: Use Franka gripper action
            
            // Lazy initialization of action clients (if not already created)
            if (!gripper_move_action_client_) {


                std::string ns = this->get_namespace();
                if (ns == "/") ns = "";

                gripper_move_action_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(
                    this, fmt::format("{}/panda_gripper/move", namespace_));
                gripper_grasp_action_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(
                    this, fmt::format("{}/panda_gripper/grasp", namespace_));
                
                // Wait for action servers
                if (!gripper_move_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
                    RCLCPP_ERROR(this->get_logger(), "Move Action server not available after waiting.");
                    return;
                }
                if (!gripper_grasp_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
                    RCLCPP_ERROR(this->get_logger(), "Grasp Action server not available after waiting.");
                    return;
                }
            }
            
            // Track gripper state for toggling
            static bool gripper_open = false;
            
            if (msg->data == 1) {
                // Button pressed: Close gripper (grasp)
                if (!gripper_open) {
                    RCLCPP_INFO(this->get_logger(), "Button pressed - closing gripper");
                    
                    franka_msgs::action::Grasp::Goal grasp_goal;
                    grasp_goal.speed = 0.5;
                    grasp_goal.force = 100.0;
                    grasp_goal.width        = 0.04;   // approximate object width
                    grasp_goal.epsilon.inner = 0.04;  // generous tolerance below
                    grasp_goal.epsilon.outer = 0.04;  // generous tolerance above
                                        
                    // Create options and set the result callback
                    auto grasp_goal_options = rclcpp_action::Client<franka_msgs::action::Grasp>::SendGoalOptions();
                    grasp_goal_options.result_callback = 
                        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>::WrappedResult& result) {
                            if (rclcpp_action::ResultCode::SUCCEEDED == result.code) {
                                RCLCPP_INFO(this->get_logger(), "Grasp succeeded");
                            } else {
                                RCLCPP_ERROR(this->get_logger(), "Grasp failed");
                            }
                        };
                    
                    gripper_grasp_action_client_->async_send_goal(grasp_goal, grasp_goal_options);
                    gripper_open = true;
                }
            } else if (msg->data == 2) {
                // Button released: Open gripper (move)
                if (gripper_open) {
                    RCLCPP_INFO(this->get_logger(), "Button released - opening gripper");
                    
                    franka_msgs::action::Move::Goal move_goal;
                    move_goal.width = 0.08;
                    move_goal.speed = 0.2;
                    
                    // Create options and set the result callback
                    auto move_goal_options = rclcpp_action::Client<franka_msgs::action::Move>::SendGoalOptions();
                    move_goal_options.result_callback = 
                        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>::WrappedResult& result) {
                            if (rclcpp_action::ResultCode::SUCCEEDED == result.code) {
                                RCLCPP_INFO(this->get_logger(), "Move succeeded");
                            } else {
                                RCLCPP_ERROR(this->get_logger(), "Move failed");
                            }
                        };
                    
                    gripper_move_action_client_->async_send_goal(move_goal, move_goal_options);
                    gripper_open = false;
                }
            }
        }

        /* original
        else {
            // Real hardware mode: Use Franka gripper action
            
            // Lazy initialization of action clients (if not already created)
            if (!gripper_move_action_client_) {


                std::string ns = this->get_namespace();
                if (ns == "/") ns = "";

                gripper_move_action_client_ = rclcpp_action::create_client<franka_msgs::action::Move>(
                    this, fmt::format("{}/panda_gripper/move", namespace_));
                gripper_grasp_action_client_ = rclcpp_action::create_client<franka_msgs::action::Grasp>(
                    this, fmt::format("{}/panda_gripper/grasp", namespace_));
                
                // Wait for action servers
                if (!gripper_move_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
                    RCLCPP_ERROR(this->get_logger(), "Move Action server not available after waiting.");
                    return;
                }
                if (!gripper_grasp_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
                    RCLCPP_ERROR(this->get_logger(), "Grasp Action server not available after waiting.");
                    return;
                }
            }
            
            // Track gripper state for toggling
            static bool gripper_open = false;
            
            if ((msg->data & 1) != 0) {
                // Button pressed: Close gripper (grasp)
                if (!gripper_open) {
                    RCLCPP_INFO(this->get_logger(), "Button pressed - closing gripper");
                    
                    franka_msgs::action::Grasp::Goal grasp_goal;
                    grasp_goal.speed = 0.15;
                    grasp_goal.force = 100.0;
                    grasp_goal.width        = 0.04;   // approximate object width
                    grasp_goal.epsilon.inner = 0.04;  // generous tolerance below
                    grasp_goal.epsilon.outer = 0.04;  // generous tolerance above
                                        
                    // Create options and set the result callback
                    auto grasp_goal_options = rclcpp_action::Client<franka_msgs::action::Grasp>::SendGoalOptions();
                    grasp_goal_options.result_callback = 
                        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Grasp>::WrappedResult& result) {
                            if (rclcpp_action::ResultCode::SUCCEEDED == result.code) {
                                RCLCPP_INFO(this->get_logger(), "Grasp succeeded");
                            } else {
                                RCLCPP_ERROR(this->get_logger(), "Grasp failed");
                            }
                        };
                    
                    gripper_grasp_action_client_->async_send_goal(grasp_goal, grasp_goal_options);
                    gripper_open = true;
                }
            } else {
                // Button released: Open gripper (move)
                if (gripper_open) {
                    RCLCPP_INFO(this->get_logger(), "Button released - opening gripper");
                    
                    franka_msgs::action::Move::Goal move_goal;
                    move_goal.width = 0.08;
                    move_goal.speed = 0.2;
                    
                    // Create options and set the result callback
                    auto move_goal_options = rclcpp_action::Client<franka_msgs::action::Move>::SendGoalOptions();
                    move_goal_options.result_callback = 
                        [this](const rclcpp_action::ClientGoalHandle<franka_msgs::action::Move>::WrappedResult& result) {
                            if (rclcpp_action::ResultCode::SUCCEEDED == result.code) {
                                RCLCPP_INFO(this->get_logger(), "Move succeeded");
                            } else {
                                RCLCPP_ERROR(this->get_logger(), "Move failed");
                            }
                        };
                    
                    gripper_move_action_client_->async_send_goal(move_goal, move_goal_options);
                    gripper_open = false;
                }
            }
        }

        */
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MEMBER VARIABLES
    // ─────────────────────────────────────────────────────────────────────────

    // ROS 2 communication
    rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
    rclcpp::Client<moveit_msgs::srv::GetPositionFK>::SharedPtr fk_client_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr haptic_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr goal_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr set_force_sub_;
    //rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr joint_wrench_sub_;
    rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr franka_state_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr haptic_force_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_joint_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_joint_vel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_joint_torque_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr scaled_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr wrench_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr ee_pos_error_pub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr button_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr gripper_pub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr mode_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr stiffness_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr robot_force_sub_;

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr force_mag_pub_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr ext_wrench_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_pub_;


    // MoveIt kinematics
    robot_model_loader::RobotModelLoaderPtr robot_model_loader_;
    moveit::core::RobotModelPtr kinematic_model_;
    moveit::core::RobotStatePtr kinematic_state_;
    const moveit::core::JointModelGroup* joint_model_group_;

    // Timing
    rclcpp::TimerBase::SharedPtr admittance_timer_;
    rclcpp::Time last_time_;

    // State
    sensor_msgs::msg::JointState latest_joint_state_;
    geometry_msgs::msg::Pose latest_haptic_pose_;
    geometry_msgs::msg::PoseStamped last_valid_pose_;
    Eigen::Vector3d actual_ee_pos_;
    bool actual_ee_pos_initialized_{false};

    // Parameters
    std::string base_frame_;
    double position_scale_x_, position_scale_y_, position_scale_z_, height_offset_;
    double valid_distance_threshold_, max_force_output_;

    // Admittance gains
    double m_;            ///< Virtual inertia M_d (kg)
    double d_;            ///< Virtual damping D_d (Ns/m) — manually set value
    double k_;            ///< Virtual stiffness K_d (N/m)
    double d_calculated;  ///< Auto-calculated damping: 2*zeta*sqrt(M*K)
    double zeta;          ///< Damping ratio (1.0 = critically damped)

    double target_gripper_width_;
    franka_msgs::action::Move::Goal last_gripper_goal_;
    bool last_gripper_goal_set_ = false;

    int mode_;
    bool use_topic_input_ = false;
    bool reachedStart = false;
    // Controller state (integrated virtual dynamics)
    Eigen::Vector3d ee_pos_;       ///< Virtual EE position (integrated)
    Eigen::Vector3d ee_vel_;       ///< Virtual EE velocity (integrated)
    Eigen::Vector3d haptic_goal_;  ///< Goal from haptic device
    Eigen::Quaterniond haptic_orientation_goal_;

    Eigen::Vector3d set_goal_;     ///< Goal from /set_goal_pose topic
    Eigen::Vector3d robot_force_;  ///< Gravity-compensated sensor force (or commanded via /robot_force)
    Eigen::Vector3d measured_force_;  ///< Raw measured external force (published to /robot/external_wrench)
    Eigen::Vector3d damping_force_;  ///< Gravity-compensated sensor force

    Eigen::Vector3d set_force_;    ///< Manually injected force
    Eigen::Vector3d gravity_vec;

    // Physical constants
    double total_mass = 1.3397;  ///< EE + payload mass (kg) for gravity compensation

        // For gripper action (real hardware mode)
    rclcpp_action::Client<franka_msgs::action::Move>::SharedPtr gripper_move_action_client_;
    rclcpp_action::Client<franka_msgs::action::Grasp>::SharedPtr gripper_grasp_action_client_;
    std::string namespace_;


};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimpleIKNode>();
    node->initializeRobotModel();  // must be called after construction (needs shared_from_this)
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
