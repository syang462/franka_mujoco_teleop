#!/usr/bin/env python3
"""Extract + plot position-error and force response from one or more bags.

Usage:
    python3 plot_bag_response.py <bag1> [bag2 ...] [--out figures]

Each bag is aligned so t=0 is the command time (first /target_pose for pose
bags, first /robot_force for force bags). All listed bags are overlaid on the
same two plots (xyz position error, xyz measured force), distinguished by line
style.

Position error = /target_pose - /robot/ee_pose (target forward-filled).
If a bag has no /target_pose, EE displacement from bag start is plotted instead.
Total force = /robot/external_wrench PLUS the commanded /robot_force
(forward-filled), so the commanded virtual force shows up in the plot.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

AXIS = {'x': '#e74c3c', 'y': '#2ecc71', 'z': '#3498db'}
DASH_STYLES = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]


def make_reader(bag_path):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    try:
        reader.open(storage_options, converter_options)
    except Exception:
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
        reader.open(storage_options, converter_options)
    return reader


def extract(bag_path):
    """Read a bag into per-topic arrays (absolute seconds)."""
    reader = make_reader(bag_path)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    want = ['/target_pose', '/robot/ee_pose', '/robot/external_wrench', '/robot_force']
    data = {t: {'t': [], 'v': []} for t in want}

    while reader.has_next():
        topic, raw, timestamp = reader.read_next()
        if topic not in want:
            continue
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(raw, msg_type)
        t = timestamp * 1e-9
        data[topic]['t'].append(t)
        if topic == '/target_pose':
            p = msg.pose.position
            data[topic]['v'].append([p.x, p.y, p.z])
        elif topic == '/robot/ee_pose':
            p = msg.pose.position
            data[topic]['v'].append([p.x, p.y, p.z])
        elif topic == '/robot/external_wrench':
            f = msg.wrench.force
            data[topic]['v'].append([f.x, f.y, f.z])
        elif topic == '/robot_force':
            data[topic]['v'].append([msg.x, msg.y, msg.z])

    arrays = {}
    for topic in want:
        arrays[topic] = {
            't': np.array(data[topic]['t']),
            'v': np.array(data[topic]['v']).reshape(-1, 3),
        }

    # Command time -> t=0
    if len(arrays['/target_pose']['t']):
        t0 = arrays['/target_pose']['t'][0]
    elif len(arrays['/robot_force']['t']):
        t0 = arrays['/robot_force']['t'][0]
    else:
        first = [arrays[t]['t'][0] for t in want if len(arrays[t]['t'])]
        t0 = min(first) if first else 0.0
    for topic in want:
        arrays[topic]['t'] = arrays[topic]['t'] - t0
    return arrays


def compute_pos_error(arrays):
    """Returns (t, err[3]) or (None, None) if no ee_pose in the bag."""
    ee = arrays['/robot/ee_pose']
    if not len(ee['t']):
        return None, None

    tgt = arrays['/target_pose']
    if len(tgt['t']):
        idx = np.searchsorted(tgt['t'], ee['t'], side='right') - 1
        valid = idx >= 0
        t = ee['t'][valid]
        idx = idx[valid]
        err = tgt['v'][idx] - ee['v'][valid]
        return t, err
    else:
        # No commanded pose: plot EE displacement from bag start instead
        return ee['t'], ee['v'] - ee['v'][0]


def compute_force(arrays):
    """Returns (t, total[3]) or (None, None) if no wrench in the bag.

    Total virtual force = measured /robot/external_wrench PLUS the commanded
    /robot_force, forward-filled onto wrench timestamps (same idea as the
    target-pose forward-fill). Before the first /robot_force message no
    command has been applied, so only the measured wrench is used there.
    Bags without /robot_force (e.g. pose-only runs) are just the wrench."""
    w = arrays['/robot/external_wrench']
    if not len(w['t']):
        return None, None

    rf = arrays['/robot_force']
    total = w['v'].copy()
    if len(rf['t']):
        idx = np.searchsorted(rf['t'], w['t'], side='right') - 1
        valid = idx >= 0
        total[valid] += rf['v'][idx[valid]]
    return w['t'], total


def main():
    parser = argparse.ArgumentParser(description='Plot pos error + force from bags')
    parser.add_argument('bags', nargs='+', help='bag directory names')
    parser.add_argument('--out', default='figures', help='output directory')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fig_err, ax_err = plt.subplots(figsize=(8, 4))
    fig_for, ax_for = plt.subplots(figsize=(8, 4))

    for i, bag in enumerate(args.bags):
        name = bag.rstrip('/').split('/')[-1]
        dash = DASH_STYLES[i % len(DASH_STYLES)]

        try:
            arrays = extract(bag)
        except Exception as e:
            print(f"!! {name}: failed to read bag ({e}), skipping.")
            continue

        t, err = compute_pos_error(arrays)
        if t is not None:
            for k, axis in enumerate('xyz'):
                ax_err.plot(t, err[:, k], color=AXIS[axis], linestyle=dash,
                            linewidth=1.5, label=name if k == 0 else None)
            print(f"{name}: peak |pos err| = {np.max(np.abs(err)):.4f} m")
        else:
            print(f"{name}: no /robot/ee_pose in bag, no pos error plotted.")

        t, f = compute_force(arrays)
        if t is not None:
            for k, axis in enumerate('xyz'):
                ax_for.plot(t, f[:, k], color=AXIS[axis], linestyle=dash,
                            linewidth=1.5, label=name if k == 0 else None)
            print(f"{name}: peak |force|   = {np.max(np.abs(f)):.3f} N")
        else:
            print(f"{name}: no /robot/external_wrench in bag, no force plotted.")

    axis_handles = [Line2D([0], [0], color=AXIS[a], lw=2, label=a) for a in 'xyz']
    bag_handles = [Line2D([0], [0], color='black', lw=2,
                          linestyle=DASH_STYLES[i % len(DASH_STYLES)],
                          label=b.rstrip('/').split('/')[-1])
                   for i, b in enumerate(args.bags)]

    for ax, handles, ylab, fname in [
            (ax_err, bag_handles, 'Position error (m)', 'pos_error.png'),
            (ax_for, bag_handles, 'Measured force (N)', 'force.png')]:
        ax.set_xlabel('Time since command (s)')
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle='--', alpha=0.4)
        leg_axis = ax.legend(handles=axis_handles, title='Axis', loc='upper right',
                             fontsize=8)
        leg_bag = ax.legend(handles=handles, title='Bag', loc='lower left',
                            bbox_to_anchor=(0.0, 1.01), ncol=len(args.bags),
                            fontsize=8, borderaxespad=0, handlelength=3.5)
        ax.add_artist(leg_axis)
        out = os.path.join(args.out, fname)
        ax.get_figure().savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")

    plt.close(fig_err)
    plt.close(fig_for)


if __name__ == '__main__':
    main()
