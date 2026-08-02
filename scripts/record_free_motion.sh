#!/bin/bash
# Record force + position-error response to a commanded target pose (free-space motion).
#
# Prereqs: run mucojo_teleop_seperate_manual.launch.py, then set controller to
# mode 1 (Toggle Mode button in the GUI or: ros2 topic pub /toggle_mode std_msgs/msg/Int32 "{data: 1}" --once).
# NOTE: /toggle_mode TOGGLES, so publishing again switches back to mode 0.

DELAY=2.0
BAG_NAME="free_motion_bag"
TX=0.5
TY=0.0
TZ=0.35

# Delete existing bag if it exists
if [ -d "$BAG_NAME" ]; then
    echo "Deleting existing bag..."
    rm -rf "$BAG_NAME"
fi

# Start recording in background
ros2 bag record /target_pose /robot/ee_pose /robot/external_wrench -o $BAG_NAME &
BAG_PID=$!

# Wait until bag folder exists (recorder is ready)
echo "Waiting for bag to initialize..."
until [ -d "${BAG_NAME}" ]; do
    sleep 0.05
done
echo "Bag confirmed recording, waiting ${DELAY}s before sending target pose..."

sleep $DELAY

# Send commanded target pose
ros2 topic pub --once /target_pose geometry_msgs/msg/PoseStamped "{
  header: {frame_id: base_link},
  pose: {position: {x: $TX, y: $TY, z: $TZ},
         orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}
}"

echo "Pose sent. Press Ctrl+C when motion is complete to stop recording."
echo "Position error = /target_pose minus /robot/ee_pose from the bag."
wait $BAG_PID
