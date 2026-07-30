import tkinter as tk
from tkinter import ttk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64, Bool


class GuiControllerNode(Node):
    def __init__(self):
        super().__init__('gui_controller')

        self.mode_pub = self.create_publisher(Int32, '/toggle_mode', 10)
        self.stiffness_pub = self.create_publisher(Float64, '/stiffness', 10)
        self.reset_pub = self.create_publisher(Bool, '/reset', 10)

        self.mode = 0

        self.root = tk.Tk()
        self.root.title('GUI Controller')
        self.root.resizable(False, False)

        mode_btn = ttk.Button(self.root, text='Toggle Mode',
                              command=self.toggle_mode)
        mode_btn.pack(padx=20, pady=(20, 5), fill='x')

        stiffness_frame = ttk.LabelFrame(self.root, text='Stiffness')
        stiffness_frame.pack(padx=20, pady=5, fill='x')

        self.stiffness_var = tk.DoubleVar(value=50.0)
        slider = ttk.Scale(stiffness_frame, from_=0.0, to=100.0,
                           orient='horizontal', variable=self.stiffness_var,
                           command=self.on_stiffness_change)
        slider.pack(padx=10, pady=10, fill='x')

        self.stiffness_label = ttk.Label(stiffness_frame,
                                         text=f'{self.stiffness_var.get():.1f}')
        self.stiffness_label.pack(pady=(0, 5))

        reset_btn = ttk.Button(self.root, text='Reset',
                               command=self.publish_reset)
        reset_btn.pack(padx=20, pady=(5, 20), fill='x')

        self.root.after(50, self.spin_rclpy)

    def toggle_mode(self):
        self.mode = 1 #- self.mode
        msg = Int32()
        msg.data = self.mode
        self.mode_pub.publish(msg)
        self.get_logger().info(f'Toggle mode -> {self.mode}')

    def on_stiffness_change(self, _=None):
        val = self.stiffness_var.get()
        self.stiffness_label.config(text=f'{val:.1f}')
        msg = Float64()
        msg.data = val
        self.stiffness_pub.publish(msg)

    def publish_reset(self):
        msg = Bool()
        msg.data = True
        self.reset_pub.publish(msg)
        self.get_logger().info('Reset published')

    def spin_rclpy(self):
        try:
            rclpy.spin_once(self, timeout_sec=0)
        except rclpy.executors.ExternalShutdownException:
            return
        self.root.after(50, self.spin_rclpy)


def main(args=None):
    rclpy.init(args=args)
    node = GuiControllerNode()
    try:
        node.root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
