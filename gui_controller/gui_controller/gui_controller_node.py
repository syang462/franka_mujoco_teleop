import tkinter as tk
from tkinter import ttk
import csv
import datetime
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64
from geometry_msgs.msg import WrenchStamped


class GuiControllerNode(Node):
    def __init__(self):
        super().__init__('gui_controller')

        self.mode_pub = self.create_publisher(Int32, '/toggle_mode', 10)
        self.stiffness_pub = self.create_publisher(Float64, '/stiffness', 10)

        self.mode = 0
        self.recording = False
        self.trial_data = []
        self.trial_start_time = None
        self.elapsed = 0.0

        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/robot/external_wrench', self.wrench_callback, 10)

        self.root = tk.Tk()
        self.root.title('GUI Controller')
        self.root.resizable(False, False)
        self.root.geometry('400x300')

        mode_btn = ttk.Button(self.root, text='Toggle Mode',
                              command=self.toggle_mode)
        mode_btn.pack(padx=20, pady=(20, 5), fill='x')

        stiffness_frame = ttk.LabelFrame(self.root, text='Stiffness')
        stiffness_frame.pack(padx=20, pady=5, fill='x')

        self.stiffness_var = tk.DoubleVar(value=200.0)
        slider = ttk.Scale(stiffness_frame, from_=5.0, to=10000.0,
                           orient='horizontal', variable=self.stiffness_var,
                           command=self.on_stiffness_change)
        slider.pack(padx=10, pady=(10, 2), fill='x')

        entry_frame = ttk.Frame(stiffness_frame)
        entry_frame.pack(pady=(0, 10))

        self.stiffness_entry = ttk.Entry(entry_frame, width=10)
        self.stiffness_entry.insert(0, f'{self.stiffness_var.get():.0f}')
        self.stiffness_entry.pack(side='left', padx=(0, 5))
        self.stiffness_entry.bind('<Return>', self.on_stiffness_entry)

        ttk.Label(entry_frame, text='N/m').pack(side='left')

        self.trial_btn = ttk.Button(self.root, text='Start Trial',
                                    command=self.toggle_trial)
        self.trial_btn.pack(padx=20, pady=(5, 2), fill='x')

        self.timer_label = ttk.Label(self.root, text='Elapsed: 0.0 s')
        self.timer_label.pack(pady=(0, 10))

        self.status_label = ttk.Label(self.root, text='')
        self.status_label.pack(pady=(0, 10))

        self.root.after(50, self.spin_rclpy)

    def toggle_mode(self):
        self.mode = 1
        msg = Int32()
        msg.data = self.mode
        self.mode_pub.publish(msg)
        self.get_logger().info(f'Toggle mode -> {self.mode}')

    def on_stiffness_change(self, _=None):
        val = self.stiffness_var.get()
        self.stiffness_entry.delete(0, 'end')
        self.stiffness_entry.insert(0, f'{val:.0f}')
        msg = Float64()
        msg.data = val
        self.stiffness_pub.publish(msg)

    def on_stiffness_entry(self, event):
        try:
            val = float(self.stiffness_entry.get())
            val = max(5.0, min(10000.0, val))
            self.stiffness_var.set(val)
            self.stiffness_entry.delete(0, 'end')
            self.stiffness_entry.insert(0, f'{val:.0f}')
            msg = Float64()
            msg.data = val
            self.stiffness_pub.publish(msg)
        except ValueError:
            pass

    def wrench_callback(self, msg):
        if not self.recording:
            return
        t = (msg.header.stamp.sec +
             msg.header.stamp.nanosec * 1e-9)
        elapsed = t - self.trial_start_time
        self.trial_data.append((
            elapsed,
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        ))

    def toggle_trial(self):
        if not self.recording:
            self.start_trial()
        else:
            self.stop_trial()

    def start_trial(self):
        self.recording = True
        self.trial_data.clear()
        now = self.get_clock().now()
        self.trial_start_time = now.seconds_nanoseconds()[0] + \
                                now.seconds_nanoseconds()[1] * 1e-9
        self.trial_btn.config(text='Stop Trial')
        self.status_label.config(text='Recording...')
        self.get_logger().info('Trial started')
        self.update_timer()

    def stop_trial(self):
        self.recording = False
        self.trial_btn.config(text='Start Trial')
        self.status_label.config(text='')
        self.get_logger().info(f'Trial stopped — {len(self.trial_data)} samples')
        max_force = None
        if self.trial_data:
            max_force = max(
                (row[1]**2 + row[2]**2 + row[3]**2) ** 0.5
                for row in self.trial_data
            )
            self.get_logger().info(f'Max force magnitude: {max_force:.3f} N')
        self.save_trial_data(max_force)

    def update_timer(self):
        if not self.recording:
            return
        now = self.get_clock().now()
        t = now.seconds_nanoseconds()[0] + now.seconds_nanoseconds()[1] * 1e-9
        elapsed = t - self.trial_start_time
        self.timer_label.config(text=f'Elapsed: {elapsed:.1f} s')
        self.root.after(100, self.update_timer)

    def save_trial_data(self, max_force=None):
        exportCSV = False
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'trial_{ts}.csv'
        if exportCSV:    
            with open(filename, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
                w.writerows(self.trial_data)
        if max_force is not None:
            self.status_label.config(
                text=f'Max |F|: {max_force:.3f} N  |  Saved: {filename}')
        else:
            self.status_label.config(text=f'Saved: {filename}')
        self.get_logger().info(f'Trial data saved to {filename}')

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
