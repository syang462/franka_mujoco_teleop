/*****************************************************************************
 ROS2 Node for OpenHaptics Device
 Publishes transform, pose, and device state to ROS2 topics
 Subscribes to force topic to control device output
*****************************************************************************/

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/wrench.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <std_msgs/msg/int32.hpp>

#include <HD/hd.h>
#include <HDU/hduError.h>
#include <HDU/hduVector.h>

#include <memory>
#include <cstring>
#include <mutex>

// Device state structure
typedef struct
{
    HDdouble forceValues[3];
    HDdouble jointTorqueValues[3];   
    HDdouble gimbalTorqueValues[3];
    HDdouble positionValues[3];
    HDdouble velocityValues[3];
    HDdouble transform[16];
    HDint currentButtons;     
    HDint lastButtons;         
} DeviceStateStruct;

// Global force command structure with thread safety
struct ForceCommand
{
    hduVector3Dd force;
    std::mutex mutex;
    bool has_data = false;
} g_forceCommand;

// Callback to get device state
HDCallbackCode HDCALLBACK GetDeviceStateCallback(void *pUserData)
{
    DeviceStateStruct *pState = (DeviceStateStruct *) pUserData;

    hdGetDoublev(HD_CURRENT_FORCE, pState->forceValues);
    hdGetDoublev(HD_CURRENT_JOINT_TORQUE, pState->jointTorqueValues);
    hdGetDoublev(HD_CURRENT_GIMBAL_TORQUE, pState->gimbalTorqueValues);
    hdGetDoublev(HD_CURRENT_POSITION, pState->positionValues);
    hdGetDoublev(HD_CURRENT_VELOCITY, pState->velocityValues);

    hdGetDoublev(HD_CURRENT_TRANSFORM, pState->transform);
    hdGetIntegerv(HD_CURRENT_BUTTONS, &pState->currentButtons);
    hdGetIntegerv(HD_LAST_BUTTONS, &pState->lastButtons);
    return HD_CALLBACK_DONE;
}

HDCallbackCode HDCALLBACK forceCallback(void *data)
{
    HDErrorInfo error;
    hduVector3Dd force;
    hduVector3Dd velocity_raw;

    // Persistent filtered velocity (keeps state across calls)
    static hduVector3Dd velocity_filtered(0.0, 0.0, 0.0);

    constexpr double ALPHA = 0.2;          // Low-pass filter coefficient
    constexpr double DAMPING_COEFF = 0.003; // Tune this!

    HHD hHD = hdGetCurrentDevice();
    hdBeginFrame(hHD);

    // Read current velocity
    hdGetDoublev(HD_CURRENT_VELOCITY, velocity_raw);

    // Apply low-pass filter
    for (int i = 0; i < 3; ++i)
    {
        velocity_filtered[i] = ALPHA * velocity_raw[i] +
                               (1.0 - ALPHA) * velocity_filtered[i];
    }

    // Read commanded force (thread-safe)
    {
        std::lock_guard<std::mutex> lock(g_forceCommand.mutex);
        if (g_forceCommand.has_data)
        {
            force = g_forceCommand.force;
        }
        else
        {
            force.set(0.0, 0.0, 0.0);
        }
    }

    // Apply damping using FILTERED velocity
    for (int i = 0; i < 3; ++i)
    {
        //force[i] -= DAMPING_COEFF * velocity_filtered[i];
    }
    
    // Send force to device
    hdSetDoublev(HD_CURRENT_FORCE, force);

    hdEndFrame(hHD);

    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        if (hduIsSchedulerError(&error))
        {
            return HD_CALLBACK_DONE;
        }
    }

    return HD_CALLBACK_CONTINUE;
}

// Convert 4x4 column-major transform to ROS transform
geometry_msgs::msg::TransformStamped matrixToTransform(
    const HDdouble* matrix, 
    const std::string& frame_id,
    const std::string& child_frame_id,
    const rclcpp::Time& timestamp)
{
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = timestamp;
    transform.header.frame_id = frame_id;
    transform.child_frame_id = child_frame_id;

    // Extract translation (last column, but column-major so indices 12,13,14)
    transform.transform.translation.x = matrix[12] / 1000.0; // Convert mm to m
    transform.transform.translation.y = matrix[13] / 1000.0;
    transform.transform.translation.z = matrix[14] / 1000.0;

    // Extract rotation matrix (first 3x3)
    tf2::Matrix3x3 rotMat(
        matrix[0], matrix[4], matrix[8],
        matrix[1], matrix[5], matrix[9],
        matrix[2], matrix[6], matrix[10]
    );

    // Convert to quaternion
    tf2::Quaternion quat;
    rotMat.getRotation(quat);
    
    transform.transform.rotation.x = quat.x();
    transform.transform.rotation.y = quat.y();
    transform.transform.rotation.z = quat.z();
    transform.transform.rotation.w = quat.w();

    return transform;
}

// Convert 4x4 column-major transform to ROS Pose
geometry_msgs::msg::Pose matrixToPose(
    const HDdouble* matrix, 
    const rclcpp::Time& timestamp)
{
    geometry_msgs::msg::Pose pose;

    // Extract translation (last column, but column-major so indices 12,13,14)
    pose.position.x = matrix[12] / 1000.0; // Convert mm to m
    pose.position.y = matrix[13] / 1000.0;
    pose.position.z = matrix[14] / 1000.0;

    // Extract rotation matrix (first 3x3)
    tf2::Matrix3x3 rotMat(
        matrix[0], matrix[4], matrix[8],
        matrix[1], matrix[5], matrix[9],
        matrix[2], matrix[6], matrix[10]
    );

    // Convert to quaternion
    tf2::Quaternion quat;
    rotMat.getRotation(quat);
    
    pose.orientation.x = quat.x();
    pose.orientation.y = quat.y();
    pose.orientation.z = quat.z();
    pose.orientation.w = quat.w();

    return pose;
}

class HapticDeviceNode : public rclcpp::Node
{
public:
    HapticDeviceNode() : Node("haptic_device_node")
    {
        // Declare parameters
        this->declare_parameter("frame_id", "world");
        this->declare_parameter("device_frame", "haptic_device");
        this->declare_parameter("publish_rate", 1000.0);
        this->declare_parameter("publish_tf", true);
        this->declare_parameter("entity_name", "haptic_device");
        this->declare_parameter("force_topic", "haptic/force_command");

        // Get parameters
        frame_id_ = this->get_parameter("frame_id").as_string();
        device_frame_ = this->get_parameter("device_frame").as_string();
        double publish_rate = this->get_parameter("publish_rate").as_double();
        publish_tf_ = this->get_parameter("publish_tf").as_bool();
        entity_name_ = this->get_parameter("entity_name").as_string();
        std::string force_topic = this->get_parameter("force_topic").as_string();

        // Create publishers
        transform_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>(
            "haptic/transform", 10);
        pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>(
            "haptic/pose", 1);
        position_pub_ = this->create_publisher<geometry_msgs::msg::Point>(
            "haptic/position", 10);
        velocity_pub_ = this->create_publisher<geometry_msgs::msg::Point>(
            "haptic/velocity", 10);
        wrench_pub_ = this->create_publisher<geometry_msgs::msg::Wrench>(
            "haptic/wrench", 10);
        joint_torque_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "haptic/joint_torques", 10);
        gimbal_torque_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "haptic/gimbal_torques", 10);
        button_state_pub_ = this->create_publisher<std_msgs::msg::Int32>(
            "haptic/buttons", 10);

        // Create force command subscriber
        force_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            force_topic, 1,
            std::bind(&HapticDeviceNode::forceCallback, this, std::placeholders::_1));

        // Initialize haptic device
        if (!initializeDevice())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize device");
            rclcpp::shutdown();
            return;
        }

        // Create timer for publishing
        auto timer_period = std::chrono::duration<double>(1.0 / publish_rate);
        timer_ = this->create_wall_timer(
            timer_period,
            std::bind(&HapticDeviceNode::publishDeviceState, this));

        RCLCPP_INFO(this->get_logger(), 
            "Haptic device node started at %.1f Hz, listening for forces on %s", 
            publish_rate, force_topic.c_str());
    }

    ~HapticDeviceNode()
    {
        cleanup();
    }

private:
    void forceCallback(const geometry_msgs::msg::Vector3::SharedPtr msg)
    {
        // Update global force command (thread-safe)
        std::lock_guard<std::mutex> lock(g_forceCommand.mutex);
        g_forceCommand.force[0] = msg->x;
        g_forceCommand.force[1] = msg->y;
        g_forceCommand.force[2] = msg->z;
        g_forceCommand.has_data = true;
    }

    bool initializeDevice()
    {
        HDErrorInfo error;
        hHD_ = hdInitDevice(HD_DEFAULT_DEVICE);
        if (HD_DEVICE_ERROR(error = hdGetError())) 
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize haptic device");
            hduPrintError(stderr, &error, "Failed to initialize haptic device");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Found device model: %s", 
                    hdGetString(HD_DEVICE_MODEL_TYPE));

        // Check calibration
        int calibrationStyle;
        hdGetIntegerv(HD_CALIBRATION_STYLE, &calibrationStyle);
        if (calibrationStyle & HD_CALIBRATION_ENCODER_RESET)
        {
            RCLCPP_INFO(this->get_logger(), "Updating calibration...");
            hdUpdateCalibration(calibrationStyle);
            if (hdCheckCalibration() == HD_CALIBRATION_OK)
            {
                RCLCPP_INFO(this->get_logger(), "Calibration complete");
            }
        }

        // Schedule force callback
        hForceCallback_ = hdScheduleAsynchronous(::forceCallback, 0, HD_MAX_SCHEDULER_PRIORITY);

        hdEnable(HD_FORCE_OUTPUT);
        hdStartScheduler();

        if (HD_DEVICE_ERROR(error = hdGetError()))
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to start scheduler");
            hduPrintError(stderr, &error, "Failed to start scheduler");
            return false;
        }

        return true;
    }

    void publishDeviceState()
    {
        // Get device state synchronously
        DeviceStateStruct deviceState;
        memset(&deviceState, 0, sizeof(DeviceStateStruct));
        hdScheduleSynchronous(GetDeviceStateCallback, &deviceState,
                            HD_DEFAULT_SCHEDULER_PRIORITY);

        auto timestamp = this->now();

        // Publish transform
        auto transform_msg = matrixToTransform(
            deviceState.transform, frame_id_, device_frame_, timestamp);
        transform_pub_->publish(transform_msg);

        // Publish pose (Ignition format)
        auto pose_msg = matrixToPose(deviceState.transform, timestamp);
        pose_pub_->publish(pose_msg);


        // Publish position
        geometry_msgs::msg::Point position_msg;
        position_msg.x = deviceState.positionValues[0] / 1000.0; // mm to m
        position_msg.y = deviceState.positionValues[1] / 1000.0;
        position_msg.z = deviceState.positionValues[2] / 1000.0;
        position_pub_->publish(position_msg);

        // Publish wrench (force + torque)
        geometry_msgs::msg::Wrench wrench_msg;
        wrench_msg.force.x = deviceState.forceValues[0];
        wrench_msg.force.y = deviceState.forceValues[1];
        wrench_msg.force.z = deviceState.forceValues[2];
        wrench_msg.torque.x = deviceState.gimbalTorqueValues[0];
        wrench_msg.torque.y = deviceState.gimbalTorqueValues[1];
        wrench_msg.torque.z = deviceState.gimbalTorqueValues[2];
        wrench_pub_->publish(wrench_msg);

        // Publish joint torques
        std_msgs::msg::Float64MultiArray joint_torque_msg;
        joint_torque_msg.data.resize(3);
        for (int i = 0; i < 3; i++)
            joint_torque_msg.data[i] = deviceState.jointTorqueValues[i];
        joint_torque_pub_->publish(joint_torque_msg);

        // Publish gimbal torques
        std_msgs::msg::Float64MultiArray gimbal_torque_msg;
        gimbal_torque_msg.data.resize(3);
        for (int i = 0; i < 3; i++)
            gimbal_torque_msg.data[i] = deviceState.gimbalTorqueValues[i];
        gimbal_torque_pub_->publish(gimbal_torque_msg);

        //Publish button states
        std_msgs::msg::Int32 button_msg;
        button_msg.data = deviceState.currentButtons;
        button_state_pub_->publish(button_msg);

        // Publish position
        geometry_msgs::msg::Point velocity_msg;
        velocity_msg.x = deviceState.velocityValues[0] / 1000.0; // mm to m
        velocity_msg.y = deviceState.velocityValues[1] / 1000.0;
        velocity_msg.z = deviceState.velocityValues[2] / 1000.0;
        velocity_pub_->publish(velocity_msg);
    }

    void cleanup()
    {
        RCLCPP_INFO(this->get_logger(), "Shutting down haptic device node");
        if (hForceCallback_ != HD_INVALID_HANDLE)
        {
            hdStopScheduler();
            hdUnschedule(hForceCallback_);
        }
        if (hHD_ != HD_INVALID_HANDLE)
        {
            hdDisableDevice(hHD_);
        }
    }

    // ROS2 members
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr transform_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr position_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr velocity_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Wrench>::SharedPtr wrench_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_torque_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gimbal_torque_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr button_state_pub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr force_sub_;
    
    // Parameters
    std::string frame_id_;
    std::string device_frame_;
    std::string entity_name_;
    bool publish_tf_;

    // Haptic device handles
    HHD hHD_ = HD_INVALID_HANDLE;
    HDSchedulerHandle hForceCallback_ = HD_INVALID_HANDLE;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HapticDeviceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}