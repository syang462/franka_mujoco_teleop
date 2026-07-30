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

    cv::namedWindow(kWindowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

    RCLCPP_INFO(this->get_logger(), "Camera viewer started, subscribing to /camera/image_raw");
    RCLCPP_INFO(this->get_logger(), "Press 'm' to toggle maximize, 'q' to quit");
  }

private:
  static constexpr const char* kWindowName = "Camera Viewer";

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");

      if (!window_init_) {
        natural_w_ = static_cast<int>(msg->width);
        natural_h_ = static_cast<int>(msg->height);
        cv::resizeWindow(kWindowName, natural_w_, natural_h_);
        window_init_ = true;
        RCLCPP_INFO(this->get_logger(), "Camera resolution: %dx%d", natural_w_, natural_h_);
      }

      cv::imshow(kWindowName, cv_ptr->image);

      int key = cv::waitKey(1);
      if (key == 'm' || key == 'M') {
        toggle_maximize();
      } else if (key == 'q' || key == 'Q') {
        rclcpp::shutdown();
      }
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  void toggle_maximize()
  {
    maximized_ = !maximized_;
    if (maximized_) {
      double aspect = static_cast<double>(natural_w_) / natural_h_;
      int w = 1280;
      int h = static_cast<int>(w / aspect);
      if (h > 960) {
        h = 960;
        w = static_cast<int>(h * aspect);
      }
      cv::resizeWindow(kWindowName, w, h);
    } else {
      cv::resizeWindow(kWindowName, natural_w_, natural_h_);
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  bool window_init_ = false;
  bool maximized_ = false;
  int natural_w_ = 0;
  int natural_h_ = 0;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraViewer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
