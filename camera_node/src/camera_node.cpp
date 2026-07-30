#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <string>
#include <vector>

class CameraNode : public rclcpp::Node
{
public:
  CameraNode() : Node("camera_node")
  {
    int req_w = this->declare_parameter("width", 1280);
    int req_h = this->declare_parameter("height", 720);

    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", 10);

    auto available = find_available_cameras();
    if (available.empty()) {
      RCLCPP_ERROR(this->get_logger(), "No USB camera found");
      return;
    }

    int camera_id;
    if (available.size() == 1) {
      camera_id = available[0];
      RCLCPP_INFO(this->get_logger(), "Found 1 camera at /dev/video%d", camera_id);
    } else {
      RCLCPP_INFO(this->get_logger(), "Multiple cameras found:");
      for (size_t i = 0; i < available.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "  [%zu] /dev/video%d", i, available[i]);
      }
      std::cout << "Enter camera index to connect: " << std::flush;
      size_t choice;
      std::cin >> choice;
      if (choice >= available.size()) {
        RCLCPP_ERROR(this->get_logger(), "Invalid selection");
        return;
      }
      camera_id = available[choice];
    }

    cap_.open(camera_id, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open camera %d", camera_id);
      return;
    }

    if (req_w > 0 && req_h > 0) {
      cap_.set(cv::CAP_PROP_FRAME_WIDTH, req_w);
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, req_h);
    }

    int cam_w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int cam_h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

    RCLCPP_INFO(this->get_logger(), "Opened camera %d at /dev/video%d — resolution: %dx%d",
                camera_id, camera_id, cam_w, cam_h);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(33),
      std::bind(&CameraNode::publish_frame, this));
  }

private:
  std::vector<int> find_available_cameras()
  {
    std::vector<int> cameras;
    for (int i = 0; i < 10; ++i) {
      std::string path = "/dev/video" + std::to_string(i);
      if (std::filesystem::exists(path)) {
        cv::VideoCapture test(i, cv::CAP_V4L2);
        if (test.isOpened()) {
          test.release();
          cameras.push_back(i);
        }
      }
    }
    return cameras;
  }

  void publish_frame()
  {
    if (!cap_.isOpened()) return;

    cv::Mat frame;
    cap_ >> frame;
    if (frame.empty()) return;

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = this->now();
    publisher_->publish(*msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  cv::VideoCapture cap_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
