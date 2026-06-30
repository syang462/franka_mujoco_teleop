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
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/int32.hpp>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <vector>
#include <Eigen/Dense>

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

std::array<int, 7> torque_sensor_adrs_;

const int N_ARM_JOINTS = 7;

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
  // update button state
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}

// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
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

    // MuJoCo fills these row-major — must match storage order
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


// Computes the external wrench at `body_id`, equivalent to Franka's O_F_ext_hat_K.
// Returns a 6-vector: [force(3); torque(3)] in the world/base frame.
//
// arm_joint_ids : joint ids for joint1..joint7, in order
// body_id       : the body whose Jacobian you want the wrench expressed at (e.g. hand/EE)
Eigen::VectorXd get_external_force(mjModel* model, mjData* data,
                                    int body_id,
                                    const std::vector<int>& arm_joint_ids,
                                    double damping = 1e-4)
{
    const int n = static_cast<int>(arm_joint_ids.size()); // 7

    // ---- 1. "Measured" joint torque from the cached torque-sensor addresses ----
    // Component [2] (z) of each sensor is the joint-axis-aligned component.
    Eigen::VectorXd tau_measured(n);
    for (int i = 0; i < n; ++i) {
        tau_measured(i) = data->sensordata[torque_sensor_adrs_[i] + 2];
    }

    // ---- 2. Model term: M(q)q̈ + C(q,q̇)q̇ + g(q), no external contact assumed ----
    Eigen::VectorXd tau_model_full(model->nv);
    mj_rne(model, data, /*flg_acc=*/1, tau_model_full.data());

    Eigen::VectorXd tau_model(n);
    for (int i = 0; i < n; ++i) {
        int dof_adr = model->jnt_dofadr[arm_joint_ids[i]];
        tau_model(i) = tau_model_full(dof_adr);
    }

    // ---- 3. External joint torque = measured − model ----
    Eigen::VectorXd tau_ext = tau_measured - tau_model;

    // ---- 4. Map to Cartesian wrench via damped pseudoinverse of J^T ----
    Eigen::MatrixXd J = get_jacobian_reduced(model, data, body_id, arm_joint_ids); // 6 x n
    Eigen::MatrixXd Jt = J.transpose();   // n x 6

    Eigen::MatrixXd JtT_Jt = Jt.transpose() * Jt; // 6x6
    Eigen::MatrixXd damped_inv = (JtT_Jt + damping * Eigen::MatrixXd::Identity(6, 6)).inverse();
    Eigen::VectorXd F_ext = damped_inv * Jt.transpose() * tau_ext; // 6x1: [force; torque]

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

std::mutex sim_mutex_;      // guards all access to mjModel*/mjData* (m, d)
std::mutex state_mutex_;    // guards latest_state_ (cheap struct copy only)
RobotState latest_state_;

rclcpp::TimerBase::SharedPtr robot_loop_timer_;


class ControllerNode : public rclcpp::Node
{
public:

    ControllerNode()
    : Node("controller")
    {
        last_time_ = this->now();
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1),
            std::bind(&ControllerNode::controlLoop, this)
        );

        robot_loop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(1),
            std::bind(&ControllerNode::robotStateLoop, this)
        );

        //Initialize Subscribers
        haptic_pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "haptic/pose", 1,
            std::bind(&ControllerNode::hapticPoseCallback, this, std::placeholders::_1));

        button_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/haptic/buttons", 10,
            std::bind(&ControllerNode::buttonCallback, this, std::placeholders::_1));

        mode_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "/toggle_mode", 10,
            [this](const std_msgs::msg::Int32::SharedPtr msg) {
                mode_ = (mode_ == 0) ? 1 : 0;
                RCLCPP_INFO(this->get_logger(), "Mode toggled → %d (%s)",
                    mode_, mode_ == 0 ? "HOME pose" : "Haptic follow");
            });

        //Initialize Publishers
        haptic_force_pub_        = this->create_publisher<geometry_msgs::msg::Vector3>("/haptic/force_command", 10);

    }

private:
  
  rclcpp::TimerBase::SharedPtr timer_;

  rclcpp::Time last_time_;

  double m_ = 10;
  double zeta = 1.5;
  double k_ = 1000;
  double d_ = 2.0 * zeta * std::sqrt(m_ * k_);
  int mode_ = 0;

  Eigen::VectorXd d_gains{{30, 30, 25, 25, 15, 10, 5}};

  Eigen::Vector3d ee_vel_ = Eigen::Vector3d::Zero();

  Eigen::VectorXd home_q{{0.0, 0.0, 0.0, -M_PI/2.0, 0.0, M_PI/2.0, M_PI/4.0}};

  Eigen::Vector3d target_pose{0.5, 0.0, 0.25};

  Eigen::Quaterniond target_orientation{0.0, 1.0, 0.0, 0.0};

  //3D systems touch specific functions
  bool useHapticGoal = false;

  double position_scale_x_ = 4.0;
  double position_scale_y_ = 4.0;
  double position_scale_z_ = 4.0;
  double height_offset_ = 0.3;
  double max_force_output_ = 5.0;

  //Publishers
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr haptic_force_pub_;

  //Subscribers and Callbacks:
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr mode_sub_;

  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr haptic_pose_sub_;

  bool inside_workspace = false;

  bool nearSwitch = false;

  geometry_msgs::msg::Vector3 wall_force;
  void hapticPoseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
        target_pose.x() = (-msg->position.z * position_scale_x_) + height_offset_;
        target_pose.y() = -msg->position.x * position_scale_y_;
        target_pose.z() = (msg->position.y * position_scale_z_) + height_offset_;

        bool nearSwitch = false;
        if(msg->position.y < -0.05 && msg->position.z < -0.05 && std::abs(msg->position.x) < 0.01){
          nearSwitch = true;
          //RCLCPP_INFO(this->get_logger(), "NEAR SWITCH, force disabled");
        }else{
          nearSwitch = false;
        }

        // Convert haptic orientation from message to Eigen
        Eigen::Quaterniond haptic_orientation(
            msg->orientation.w,
            msg->orientation.x,
            msg->orientation.y,
            msg->orientation.z
        );

        // Workspace boundaries (robot frame)
        const double x_min=0.2, x_max=0.6;
        const double y_min=-0.25, y_max=0.25;
        const double z_min=0.13, z_max=0.70;

        // Check if target_pose is inside workspace boundaries
        inside_workspace = 
            (target_pose.x() >= x_min && target_pose.x() <= x_max) &&
            (target_pose.y() >= y_min && target_pose.y() <= y_max) &&
            (target_pose.z() >= z_min && target_pose.z() <= z_max);

        if(!nearSwitch){
          const double k_wall = 100.0;  // wall stiffness (N/m) — tune to feel
          Eigen::Vector3d boundary_force = Eigen::Vector3d::Zero();

          // X axis
          if      (target_pose.x() < x_min) boundary_force.x() =  k_wall * (x_min - target_pose.x());
          else if (target_pose.x() > x_max) boundary_force.x() = -k_wall * (target_pose.x() - x_max);

          // Y axis
          if      (target_pose.y() < y_min) boundary_force.y() =  k_wall * (y_min - target_pose.y());
          else if (target_pose.y() > y_max) boundary_force.y() = -k_wall * (target_pose.y() - y_max);

          // Z axis
          if      (target_pose.z() < z_min) boundary_force.z() =  k_wall * (z_min - target_pose.z());
          else if (target_pose.z() > z_max) boundary_force.z() = -k_wall * (target_pose.z() - z_max);

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
              wall_force = force_out;
          }
        }

        // Hard clamp after force is computed (so penetration depth is accurate)
        target_pose.x() = std::clamp(target_pose.x(), x_min, x_max);
        target_pose.y() = std::clamp(target_pose.y(), y_min, y_max);
        target_pose.z() = std::clamp(target_pose.z(), z_min, z_max);

        Eigen::Quaterniond rot_1(Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond rot_0(Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond base_down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
        target_orientation = (rot_1 * haptic_orientation * rot_0 * base_down).normalized();
  }

  void robotForceToHapticForce(Eigen::VectorXd F_ext)
    {

        F_ext *= 0.3;
        
        Eigen::Vector3d robot_force_in(-F_ext.x(), -F_ext.y(), -F_ext.z());
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

        // ── Clamp magnitude while preserving direction ────────────────────────
        double magnitude = std::sqrt(
            force_out.x * force_out.x +
            force_out.y * force_out.y +
            force_out.z * force_out.z);
        if (magnitude > max_force_output_) {
            double scale = max_force_output_ / magnitude;
            force_out.x *= scale; force_out.y *= scale; force_out.z *= scale;
        }

        if (!FORCE_OUTPUT_ENABLED || nearSwitch) force_out.x = force_out.y = force_out.z = 0.0;
        
        bool renderWalls = false;
        if(renderWalls){
          if(inside_workspace){
            haptic_force_pub_->publish(force_out); 
          }else{
            haptic_force_pub_->publish(wall_force); 
          }
        }else{
          haptic_force_pub_->publish(force_out); 
        }
        
        
    }

  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr button_sub_;
  void buttonCallback(const std_msgs::msg::Int32::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(sim_mutex_);
      if (msg->data == 1) {
          d->ctrl[7] = 0.0;
      } else if (msg->data == 2) {
          d->ctrl[7] = 255.0;
      }
  }
  
  void robotStateLoop() {
      RobotState new_state;

      std::vector<int> joint_ids_local = {
          joint_ids[0], joint_ids[1], joint_ids[2], joint_ids[3],
          joint_ids[4], joint_ids[5], joint_ids[6]
      };

      {
          std::lock_guard<std::mutex> lock(sim_mutex_);

          mj_step(m, d);

          new_state.q    = get_current_joint_positions(m, d, joint_ids_local);
          new_state.qdot = get_current_joint_velocities(m, d, joint_ids_local);
          new_state.jacobian = get_jacobian_reduced(m, d, ee_body_id, joint_ids_local);
          new_state.external_force = get_external_force(m, d, ee_body_id, joint_ids_local);

          get_end_effector_pose(d, ee_body_id, new_state.ee_pos, new_state.ee_orientation);
          get_end_effector_velocity(d, ee_body_id, new_state.ee_vel, new_state.ee_angular_vel);
      }

      new_state.stamp = this->now();
      new_state.valid = true;

      {
          std::lock_guard<std::mutex> lock(state_mutex_);
          latest_state_ = new_state;
      }

      
  }


  void controlLoop(){
    auto now = this->now();
    double dt = (now - last_time_).seconds();
    last_time_ = now;
    if (dt <= 0.0 || dt > 0.1) {
        return;
    }

    RobotState state;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state = latest_state_;
    }

    if (!state.valid) {
        RCLCPP_WARN(this->get_logger(), "No robot state yet, skipping control tick");
        return;
    }

    //RCLCPP_INFO(this->get_logger(), "DT: %.4f", dt);

    Eigen::VectorXd q_dot(7);
    Eigen::VectorXd tau;

    if (mode_ == 0) {
        const double k_home = 2.0;
        const double max_home_vel = 0.5;

        Eigen::VectorXd target_qdot_m1(7);
        for (int i = 0; i < 7; i++) {
            double err = home_q[i] - state.q(i);
            target_qdot_m1(i) = std::clamp(k_home * err, -max_home_vel, max_home_vel);
        }

        q_dot = target_qdot_m1 - state.qdot;
        tau = q_dot.cwiseProduct(d_gains);

        {
            std::lock_guard<std::mutex> lock(sim_mutex_);
            for (int i = 0; i < N_ARM_JOINTS; i++) {
                d->ctrl[actuator_ids[i]] = tau(i);
            }
        }

        //RCLCPP_INFO(this->get_logger(), "q_dot norm: %.4f", q_dot.norm());
        //RCLCPP_INFO(this->get_logger(), "tau norm: %.4f", tau.norm());
        return;
    }

    Eigen::Vector3d error = state.ee_pos - target_pose;

    Eigen::Vector3d effective_force = Eigen::Vector3d::Zero();
    // state.external_force.head<3>() is sitting right here whenever you're
    // ready to feed measured contact force into this admittance law.

    Eigen::Vector3d acceleration = (effective_force - (d_ * ee_vel_) - (k_ * error)) / m_;
    ee_vel_ += acceleration * dt;

    Eigen::Quaterniond q_err = target_orientation * state.ee_orientation.inverse();
    q_err.normalize();

    //RCLCPP_INFO(this->get_logger(), "Cartesian error: [%.4f, %.4f, %.4f]", error.x(), error.y(), error.z());
    //RCLCPP_INFO(this->get_logger(), "Orient error: [%.4f, %.4f, %.4f, %.4f]", q_err.w(), q_err.x(), q_err.y(), q_err.z());

    if (q_err.w() < 0.0) q_err.coeffs() = -q_err.coeffs();

    double k_orient = 50;
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
    tau = q_dot.cwiseProduct(d_gains);
    
    robotForceToHapticForce(state.external_force.head<3>());
    //RCLCPP_INFO_STREAM(this->get_logger(), "external_force head<3>: " << state.external_force.head<3>().transpose());

    {
        std::lock_guard<std::mutex> lock(sim_mutex_);
        for (int i = 0; i < N_ARM_JOINTS; i++) {
            d->ctrl[actuator_ids[i]] = tau[i];
        }
    }

      //RCLCPP_INFO(this->get_logger(), "tau norm: %.4f", tau.norm());
  }
};


// main function
int main(int argc, const char** argv) {
  // check command-line arguments
  if (argc!=2) {
    std::printf(" USAGE:  basic modelfile\n");
    return EXIT_FAILURE;
  }

  // Initialize ROS 2
  rclcpp::init(argc, argv);

  // load and compile model
  char error[1000] = "Could not load binary model";
  if (std::strlen(argv[1])>4 && !std::strcmp(argv[1]+std::strlen(argv[1])-4, ".mjb")) {
    m = mj_loadModel(argv[1], 0);
  } else {
    m = mj_loadXML(argv[1], 0, error, 1000);
  }
  if (!m) {
    mju_error("Load model error: %s", error);
  }

  // make data
  d = mj_makeData(m);

  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  // create window, make OpenGL context current, request v-sync
  GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  initialize_robot();

  // Create the controller node
  auto node = std::make_shared<ControllerNode>();

  // ========== MODIFIED: Spin ROS in separate thread ==========
  std::atomic<bool> ros_spinning{true};
  
  // Create a multi-threaded executor for better performance
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  
  // Launch ROS spinning in a separate thread
  std::thread ros_spinner([&executor, &ros_spinning]() {
    while (ros_spinning.load()) {
      // spin_some with a short timeout to allow clean shutdown
      executor.spin_some(std::chrono::milliseconds(10));
      // Small sleep to prevent CPU starvation
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  // ========== END MODIFICATION ==========

  // run main loop, target real-time simulation and 60 fps rendering
  while (!glfwWindowShouldClose(window) && rclcpp::ok()) {
    // advance interactive simulation for 1/60 sec
    //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
    //  this loop will finish on time for the next frame to be rendered at 60 fps.
    //  Otherwise add a cpu timer and exit this loop when it is time to render.

    // I moved mujoco step to the control loop

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
  }

  // ========== ADDED: Clean shutdown of ROS spinner ==========
  // Signal the ROS spinner thread to stop
  ros_spinning = false;
  
  // Wait for the ROS spinner thread to finish
  if (ros_spinner.joinable()) {
    ros_spinner.join();
  }
  
  // Remove the node from executor
  executor.remove_node(node);
  // ========== END ADDITION ==========

  //free visualization storage
  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  // free MuJoCo model and data
  mj_deleteData(d);
  mj_deleteModel(m);

  // Shutdown ROS 2
  rclcpp::shutdown();

  // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return EXIT_SUCCESS;
}