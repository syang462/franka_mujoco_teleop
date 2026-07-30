#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class CameraViewer : public rclcpp::Node
{
public:
  CameraViewer() : Node("camera_viewer")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/image_raw", 10,
      std::bind(&CameraViewer::image_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Camera viewer started, subscribing to /camera/image_raw");
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
      cv::imshow("Camera Viewer", cv_ptr->image);
      cv::waitKey(1);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraViewer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
