// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// This node owns the MuJoCo simulation ("hardware interface"). It:
//   - steps physics and applies commanded joint velocities (converted to
//     torque) coming in on /target_joint_velocities
//   - publishes the robot state the controller node needs on topics:
//       /robot/joint_states     (sensor_msgs/JointState: q, qdot)
//       /robot/ee_pose          (geometry_msgs/PoseStamped)
//       /robot/jacobian         (std_msgs/Float64MultiArray, 6x7 row-major)
//       /robot/external_wrench  (geometry_msgs/WrenchStamped)
//   - renders the scene
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <vector>
#include <Eigen/Dense>

#include <random>

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

//robot properties
std::vector<int> actuator_ids;
std::vector<int> joint_ids;
int ee_body_id;

int gripper_actuator_id = -1;
int gripper_finger_joint_id = -1;

std::array<int, 7> torque_sensor_adrs_;

const int N_ARM_JOINTS = 7;

std::mutex sim_mutex_;      // guards all access to mjModel*/mjData* (m, d)

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
    std::lock_guard<std::mutex> lock(sim_mutex_);
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);
  glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  if (!button_left && !button_middle && !button_right) {
    return;
  }
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  int width, height;
  glfwGetWindowSize(window, &width, &height);

  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }
  mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

void initialize_robot(){
    actuator_ids.clear();
    joint_ids.clear();

    for (int i = 1; i <= N_ARM_JOINTS; i++){
        std::string a_name = "actuator" + std::to_string(i);
        int aid = mj_name2id(m,mjOBJ_ACTUATOR,a_name.c_str());
        if (aid == -1) mju_error("Actuator %d not found", aid);
        actuator_ids.push_back(aid);

        std::string j_name = "joint" + std::to_string(i);
        int jid = mj_name2id(m,mjOBJ_JOINT,j_name.c_str());
        if (jid == -1) mju_error("Joint %s not found", j_name.c_str());
        joint_ids.push_back(jid);
    }

    ee_body_id = mj_name2id(m,mjOBJ_BODY,"hand");
    if(ee_body_id == -1) mju_error("End effector body '%s' not found", "hand");

    for (int i = 0; i < 7; ++i) {
        int sid = mj_name2id(m, mjOBJ_SENSOR, ("j" + std::to_string(i+1) + "_torque_sensor").c_str());
        torque_sensor_adrs_[i] = m->sensor_adr[sid];
    }

    // in initialize_robot():
    gripper_actuator_id = mj_name2id(m, mjOBJ_ACTUATOR, "actuator8");
    if (gripper_actuator_id == -1) mju_error("Gripper actuator not found");

    gripper_finger_joint_id = mj_name2id(m, mjOBJ_JOINT, "finger_joint1");  // one finger; width = 2x this
    if (gripper_finger_joint_id == -1) mju_error("Gripper finger joint not found");

    m->opt.timestep = 0.001;
}

Eigen::VectorXd get_current_joint_positions(mjModel* model, mjData* data, const std::vector<int>& joint_ids)
{
    Eigen::VectorXd qpos(joint_ids.size());
    for(size_t i = 0; i < joint_ids.size(); i++){
        int jid = joint_ids[i];
        int dof_addr = model->jnt_qposadr[jid];
        qpos(i) = data->qpos[dof_addr];
    }
    return qpos;
}

Eigen::VectorXd get_current_joint_velocities(mjModel* model,mjData* data, const std::vector<int>& joint_ids)
{
    Eigen::VectorXd qvel(joint_ids.size());
    for(size_t i = 0; i < joint_ids.size(); i++){
        int jid = joint_ids[i];
        int dof_addr = model->jnt_dofadr[jid];
        qvel(i) = data->qvel[dof_addr];
    }
    return qvel;
}

void get_end_effector_pose(mjData* data,int body_id, Eigen::Vector3d& position, Eigen::Quaterniond& orientation)
{
    position <<
        data->xpos[3*body_id],
        data->xpos[3*body_id + 1],
        data->xpos[3*body_id + 2];

    orientation = Eigen::Quaterniond(
        data->xquat[4*body_id],
        data->xquat[4*body_id + 1],
        data->xquat[4*body_id + 2],
        data->xquat[4*body_id + 3]
    );
}

void get_end_effector_velocity(mjData* data, int body_id, Eigen::Vector3d& linear_vel, Eigen::Vector3d& angular_vel)
{
    linear_vel <<
        data->cvel[6*body_id + 0],
        data->cvel[6*body_id + 1],
        data->cvel[6*body_id + 2];

    angular_vel <<
        data->cvel[6*body_id + 3],
        data->cvel[6*body_id + 4],
        data->cvel[6*body_id + 5];
}

Eigen::MatrixXd get_jacobian_reduced(mjModel* model, mjData* data, int body_id, const std::vector<int>& joint_ids)
{
    int nv = model->nv;

    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp(3, nv);
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacr(3, nv);

    mj_jacBody(model, data, jacp.data(), jacr.data(), body_id);

    Eigen::MatrixXd jac_full(6, nv);
    jac_full.block(0, 0, 3, nv) = jacp;
    jac_full.block(3, 0, 3, nv) = jacr;

    Eigen::MatrixXd jac_reduced(6, joint_ids.size());
    for (size_t i = 0; i < joint_ids.size(); i++) {
        int jid = joint_ids[i];
        int dof_index = model->jnt_dofadr[jid];
        jac_reduced.col(i) = jac_full.col(dof_index);
    }
    return jac_reduced;
}


Eigen::VectorXd get_external_force(
    mjModel* model,
    mjData* data,
    int body_id,
    const std::vector<int>& arm_joint_ids,
    const Eigen::VectorXd& noise_std,
    const Eigen::VectorXd& torque_thresholds,
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr mode_pub,
    double damping = 1e-4)
{
    const int n = static_cast<int>(arm_joint_ids.size()); // 7

    Eigen::VectorXd tau_measured(n);

    for (int i = 0; i < n; ++i) {
        tau_measured(i) =
            data->sensordata[torque_sensor_adrs_[i] + 2];
    }

    // ---------------------------------------------------------
    // Joint velocities (needed for friction model)
    // ---------------------------------------------------------
    Eigen::VectorXd qvel(n);
    for (int i = 0; i < n; ++i) {
        int dof_adr = model->jnt_dofadr[arm_joint_ids[i]];
        qvel(i) = data->qvel[dof_adr];
    }

    // ---------------------------------------------------------
    // "True" friction — physically present, corrupts the sensor
    // reading exactly like a real robot's actual joint friction.
    // Coulomb (stiction) + viscous terms. tanh() gives a smooth
    // sign() so it's well-behaved near qvel == 0.
    // ---------------------------------------------------------
// Physically-motivated guess: friction scaled from each joint's rated
// torque (a real spec, not assumed), using typical harmonic-drive
// friction fractions (~1.5% Coulomb, ~4x viscous) as an engineering
// rule of thumb. NOT calibrated to a real Panda — see caveat below.
static constexpr double kFrictionScale = 0.50; // tune this

static const Eigen::VectorXd true_coulomb =
    kFrictionScale * (Eigen::VectorXd(7) << 1.3, 1.3, 1.3, 1.3, 0.18, 0.18, 0.18).finished();
static const Eigen::VectorXd true_viscous =
    kFrictionScale * (Eigen::VectorXd(7) << 5.2, 5.2, 5.2, 5.2, 0.72, 0.72, 0.72).finished();
static constexpr double kStictionSharpness = 60.0; // ~3 / 0.05 rad/s assumed breakaway velocity


    Eigen::VectorXd tau_friction_true(n);
    for (int i = 0; i < n; ++i) {
        tau_friction_true(i) =
            true_coulomb(i) * std::tanh(kStictionSharpness * qvel(i)) +
            true_viscous(i) * qvel(i);
    }

    // Physically, friction acts as a drag torque opposing motion, so it
    // reduces the torque the sensor sees relative to pure rigid-body demand.
    tau_measured -= tau_friction_true;

    // ---------------------------------------------------------
    // Apply sensor noise
    // ---------------------------------------------------------
    static std::random_device rd;
    static std::mt19937 generator(rd());

    for (int i = 0; i < n; ++i) {
        std::normal_distribution<double> noise_distribution(
            0.0,
            noise_std(i)
        );

        tau_measured(i) += noise_distribution(generator);
    }

    // ---------------------------------------------------------
    // Model torque (rigid-body dynamics only — no friction)
    // ---------------------------------------------------------
    Eigen::VectorXd tau_model_full(model->nv);

    mj_rne(
        model,
        data,
        /*flg_acc=*/1,
        tau_model_full.data()
    );

    Eigen::VectorXd tau_model(n);

    for (int i = 0; i < n; ++i) {
        int dof_adr = model->jnt_dofadr[arm_joint_ids[i]];
        tau_model(i) = tau_model_full(dof_adr);
    }

    // ---------------------------------------------------------
    // "Estimated" friction — what the robot's internal model
    // *believes* friction is. Deliberately imperfect (scaled off
    // true values) so it doesn't fully cancel tau_friction_true
    // above. This mismatch is what leaks into tau_ext, just like
    // a real robot's onboard friction compensation never being
    // perfectly calibrated.
    // ---------------------------------------------------------
    static const double kModelError = 0.90; // model believes friction is 75% of true value //adjist up to maybe 1.1 as well
    Eigen::VectorXd tau_friction_estimated = kModelError * tau_friction_true;

    tau_model -= tau_friction_estimated;

    // ---------------------------------------------------------
    // External torque
    // ---------------------------------------------------------
    Eigen::VectorXd tau_ext = tau_measured - tau_model;

    // ---------------------------------------------------------
    // Check torque thresholds
    // ---------------------------------------------------------
    static bool torque_threshold_exceeded = false;

    bool threshold_exceeded = false;

    for (int i = 0; i < n; ++i) {
        if (std::abs(tau_ext(i)) > torque_thresholds(i)) {
            threshold_exceeded = true;

            std::cout
                << "Torque threshold exceeded on J"
                << (i + 1)
                << ": "
                << tau_ext(i)
                << " Nm (threshold = "
                << torque_thresholds(i)
                << " Nm)"
                << std::endl;
        }
    }

    if (threshold_exceeded && !torque_threshold_exceeded) {
        std_msgs::msg::Int32 msg;
        msg.data = 1;
        mode_pub->publish(msg);
        torque_threshold_exceeded = true;
    }
    else if (!threshold_exceeded) {
        torque_threshold_exceeded = false;
    }

    // ---------------------------------------------------------
    // External wrench
    // ---------------------------------------------------------
    Eigen::MatrixXd J =
        get_jacobian_reduced(model, data, body_id, arm_joint_ids);

    Eigen::MatrixXd Jt = J.transpose();
    Eigen::MatrixXd JtT_Jt = Jt.transpose() * Jt;
    Eigen::MatrixXd damped_inv =
        (JtT_Jt + damping * Eigen::MatrixXd::Identity(6, 6)).inverse();

    Eigen::VectorXd F_ext = damped_inv * Jt.transpose() * tau_ext;

    return F_ext;
}

struct RobotState {
    bool valid = false;
    Eigen::VectorXd q;                  // joint positions (7)
    Eigen::VectorXd qdot;               // joint velocities (7)
    Eigen::MatrixXd jacobian;           // 6x7
    Eigen::VectorXd external_force;     // 6: [force; torque] at EE
    Eigen::Vector3d ee_pos;
    Eigen::Quaterniond ee_orientation;
    Eigen::Vector3d ee_vel;
    Eigen::Vector3d ee_angular_vel;
    rclcpp::Time stamp;
};

rclcpp::TimerBase::SharedPtr robot_loop_timer_;

class HardwareNode : public rclcpp::Node
{
public:

    HardwareNode()
    : Node("sim_hardware_node")
    {
        robot_loop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1),
            std::bind(&HardwareNode::robotStateLoop, this)
        );

        button_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/haptic/buttons", 10,
            std::bind(&HardwareNode::buttonCallback, this, std::placeholders::_1));

        velocity_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
          "/target_joint_velocities", 1,
          [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
            if (static_cast<int>(msg->data.size()) != N_ARM_JOINTS) {
              return;
            }
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            for (int i = 0; i < N_ARM_JOINTS; ++i) {
              target_joint_vel_(i) = msg->data[i];
            }
          });

        // Robot-state publishers — this is the only interface the
        // controller node needs to see.
        joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/robot/joint_states", 1);
        ee_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/robot/ee_pose", 1);
        jacobian_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "/robot/jacobian", 1);
        wrench_pub_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
            "/robot/external_wrench", 1);

        mode_pub_ = this->create_publisher<std_msgs::msg::Int32>("/toggle_mode", 10);

    }

private:

    Eigen::VectorXd d_gains{{50, 50, 45, 45, 15, 10, 5}};

    const Eigen::VectorXd torque_thresholds =
    (Eigen::VectorXd(7) << 
        20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0).finished();

    float noiseVal = 0.01;
    Eigen::VectorXd noise_std{{noiseVal, noiseVal, noiseVal, noiseVal, noiseVal, noiseVal, noiseVal}};

    std::mutex cmd_mutex_;
    Eigen::VectorXd target_joint_vel_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr button_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr velocity_sub_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ee_pose_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr jacobian_pub_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_pub_;

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr mode_pub_;


    static constexpr double kGripperCtrlMax   = 255.0;
    static constexpr double kMaxGripperWidth  = 0.08;   // meters, full open — calibrate to your MJCF

    struct GripperCommand {
        double target_width   = kMaxGripperWidth;
        double speed           = 0.5;   // m/s ramp rate on the commanded setpoint
        double force            = 100.0;  // N, applied as a runtime forcerange clamp
        double epsilon_inner  = 0.005;
        double epsilon_outer  = 0.005;
        bool   check_epsilon    = false; // true = grasp semantics, false = plain move
        bool open = false;
    };

    std::mutex gripper_cmd_mutex_;
    GripperCommand gripper_cmd_;
    double gripper_commanded_width_ = kMaxGripperWidth;  // ramped setpoint
    double gripper_last_width_ = kMaxGripperWidth;
    
    bool gripper_settled_ = true;
    double gripper_stall_timer_ = 0.0;               // seconds since motion last exceeded threshold
    static constexpr double kMoveThreshold = 1e-5;   // m, per-tick displacement considered "moving"
    static constexpr double kStallDuration = 0.5;    // s, must be still this long to count as stalled

    int last_button_state_ = 0;

    void buttonCallback(const std_msgs::msg::Int32::SharedPtr msg) {
        if (msg->data == last_button_state_) {
            return;  // still held / no change — don't re-issue the command
        }
        last_button_state_ = msg->data;

        if (msg->data == 1) {
            commandGripper(0.0, 0.1, 200.0, 0.005, 0.005, true, false);
        } else if (msg->data == 2) {
            commandGripper(kMaxGripperWidth, 0.5, 200.0, 0.0, 0.0, false, true);
        }
    }

    void commandGripper(double target_width, double speed, double force,
                        double eps_inner, double eps_outer, bool check_epsilon, bool open) {
        std::lock_guard<std::mutex> lock(gripper_cmd_mutex_);
        gripper_cmd_.target_width  = std::clamp(target_width, 0.0, kMaxGripperWidth);
        gripper_cmd_.speed          = speed;
        gripper_cmd_.force           = force;
        gripper_cmd_.epsilon_inner = eps_inner;
        gripper_cmd_.epsilon_outer = eps_outer;
        gripper_cmd_.check_epsilon   = check_epsilon;
        gripper_cmd_.open = open;
        gripper_settled_ = false;
        gripper_stall_timer_ = 0.0;   // reset on every new command
        // Honor the requested force as an actual force limit on the actuator.
        // mjModel fields are writable at runtime.
        m->actuator_forcerange[2 * gripper_actuator_id + 0] = -force;
        m->actuator_forcerange[2 * gripper_actuator_id + 1] =  force;
    }

    void publishRobotState(const RobotState& state) {
        auto stamp = state.stamp;

        sensor_msgs::msg::JointState js;
        js.header.stamp = stamp;
        js.position.resize(N_ARM_JOINTS);
        js.velocity.resize(N_ARM_JOINTS);
        for (int i = 0; i < N_ARM_JOINTS; ++i) {
            js.position[i] = state.q(i);
            js.velocity[i] = state.qdot(i);
        }
        joint_state_pub_->publish(js);

        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = stamp;
        pose.pose.position.x = state.ee_pos.x();
        pose.pose.position.y = state.ee_pos.y();
        pose.pose.position.z = state.ee_pos.z();
        pose.pose.orientation.w = state.ee_orientation.w();
        pose.pose.orientation.x = state.ee_orientation.x();
        pose.pose.orientation.y = state.ee_orientation.y();
        pose.pose.orientation.z = state.ee_orientation.z();
        ee_pose_pub_->publish(pose);

        // Flatten 6x7 jacobian row-major: data[row*7 + col]
        std_msgs::msg::Float64MultiArray jac_msg;
        jac_msg.data.resize(6 * N_ARM_JOINTS);
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < N_ARM_JOINTS; ++c) {
                jac_msg.data[r * N_ARM_JOINTS + c] = state.jacobian(r, c);
            }
        }
        jacobian_pub_->publish(jac_msg);

        geometry_msgs::msg::WrenchStamped wrench;
        wrench.header.stamp = stamp;
        wrench.wrench.force.x  = state.external_force(0);
        wrench.wrench.force.y  = state.external_force(1);
        wrench.wrench.force.z  = state.external_force(2);
        wrench.wrench.torque.x = state.external_force(3);
        wrench.wrench.torque.y = state.external_force(4);
        wrench.wrench.torque.z = state.external_force(5);
        wrench_pub_->publish(wrench);
    }

    void robotStateLoop() {
        RobotState new_state;

        Eigen::VectorXd cmd_local;
        {
            std::lock_guard<std::mutex> lock(cmd_mutex_);
            cmd_local = target_joint_vel_;
        }

        std::vector<int> joint_ids_local = {
            joint_ids[0], joint_ids[1], joint_ids[2], joint_ids[3],
            joint_ids[4], joint_ids[5], joint_ids[6]
        };

        {
            std::lock_guard<std::mutex> lock(sim_mutex_);

            // "Hardware interface" write: apply the last commanded joint
            // velocities (converted to torque via d_gains) before stepping.
            Eigen::VectorXd tau = cmd_local.cwiseProduct(d_gains);
            for (int i = 0; i < N_ARM_JOINTS; i++) {
                d->ctrl[actuator_ids[i]] = tau[i];
            }

            stepGripper();   // <-- add this

            mj_step(m, d);

            new_state.q    = get_current_joint_positions(m, d, joint_ids_local);
            new_state.qdot = get_current_joint_velocities(m, d, joint_ids_local);
            new_state.jacobian = get_jacobian_reduced(m, d, ee_body_id, joint_ids_local);
            new_state.external_force = get_external_force(m, d, ee_body_id, joint_ids_local, noise_std, torque_thresholds, mode_pub_);

            get_end_effector_pose(d, ee_body_id, new_state.ee_pos, new_state.ee_orientation);
            get_end_effector_velocity(d, ee_body_id, new_state.ee_vel, new_state.ee_angular_vel);
        }

        new_state.stamp = this->now();
        new_state.valid = true;

        // Publish outside sim_mutex_ so DDS calls never hold up mj_step.
        publishRobotState(new_state);
    }

    void stepGripper() {
        // Called with sim_mutex_ already held.
        GripperCommand cmd;
        {
            std::lock_guard<std::mutex> lock(gripper_cmd_mutex_);
            cmd = gripper_cmd_;
        }

        const double dt = m->opt.timestep;
        double current_width = 2.0 * d->qpos[m->jnt_qposadr[gripper_finger_joint_id]];

        double max_step = cmd.speed * dt;

        double err;
        if(cmd.open){
          err = cmd.target_width - gripper_commanded_width_;
        }else{
          err = -gripper_commanded_width_;
        }
        
        gripper_commanded_width_ += std::clamp(err, -max_step, max_step);

        double ctrl = std::clamp(gripper_commanded_width_ / kMaxGripperWidth * kGripperCtrlMax,
                                  0.0, kGripperCtrlMax);
        d->ctrl[gripper_actuator_id] = ctrl;


        bool moved_this_tick = std::abs(current_width - gripper_last_width_) >= kMoveThreshold;
        gripper_last_width_ = current_width;

        gripper_stall_timer_ = moved_this_tick ? 0.0 : (gripper_stall_timer_ + dt);
        bool stalled = gripper_stall_timer_ >= kStallDuration;

        if (!gripper_settled_ && (stalled)) {
            gripper_settled_ = true;
            if (cmd.check_epsilon) {
                bool success = current_width >= (cmd.target_width - cmd.epsilon_inner) &&
                                current_width <= (cmd.target_width + cmd.epsilon_outer);
                const char* cause = stalled ? "stalled" : "ramp reached target";
                RCLCPP_INFO(this->get_logger(),
                    "Grasp %s (width=%.4f target=%.4f, cause=%s, stall_timer=%.3f)",
                    success ? "succeeded" : "failed", current_width, cmd.target_width,
                    cause, gripper_stall_timer_);
            }
        }
    }
};

// main function
int main(int argc, const char** argv) {
  if (argc!=2) {
    std::printf(" USAGE:  basic modelfile\n");
    return EXIT_FAILURE;
  }

  rclcpp::init(argc, argv);

  char error[1000] = "Could not load binary model";
  if (std::strlen(argv[1])>4 && !std::strcmp(argv[1]+std::strlen(argv[1])-4, ".mjb")) {
    m = mj_loadModel(argv[1], 0);
  } else {
    m = mj_loadXML(argv[1], 0, error, 1000);
  }
  if (!m) {
    mju_error("Load model error: %s", error);
  }

  d = mj_makeData(m);

  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  GLFWwindow* window = glfwCreateWindow(1280, 720, "Teleop Sim.", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  mjv_defaultCamera(&cam);
  //cam.type = mjCAMERA_FIXED;
  cam.fixedcamid = mj_name2id(m, mjOBJ_CAMERA, "teleop_camera");

  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  initialize_robot();

  auto node = std::make_shared<HardwareNode>();

  std::atomic<bool> ros_spinning{true};

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread ros_spinner([&executor, &ros_spinning]() {
    while (ros_spinning.load()) {
      executor.spin_some(std::chrono::milliseconds(10));
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });

  while (!glfwWindowShouldClose(window) && rclcpp::ok()) {
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // Locked so we never read `d` while robotStateLoop is mid mj_step /
    // mid state-extraction on another thread — this is what fixes the jitter.
    {
        std::lock_guard<std::mutex> lock(sim_mutex_);
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    }
    mjr_render(viewport, &scn, &con);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  ros_spinning = false;
  if (ros_spinner.joinable()) {
    ros_spinner.join();
  }
  executor.remove_node(node);

  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  mj_deleteData(d);
  mj_deleteModel(m);

  rclcpp::shutdown();

#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return EXIT_SUCCESS;
}