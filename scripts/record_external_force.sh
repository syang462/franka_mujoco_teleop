#!/bin/bash
# Record force + position-error response to a commanded external force.
#
# Prereqs: run mucojo_teleop_seperate_manual.launch.py, then set controller to
# mode 1 (Toggle Mode button in the GUI or: ros2 topic pub /toggle_mode std_msgs/msg/Int32 "{data: 1}" --once).
# NOTE: /toggle_mode TOGGLES, so publishing again switches back to mode 0.

DELAY=2.0
BAG_NAME="external_force_bag"
FX=0.0
FY=10.0
FZ=0.0

# Delete existing bag if it exists
if [ -d "$BAG_NAME" ]; then
    echo "Deleting existing bag..."
    rm -rf "$BAG_NAME"
fi

# Start recording in background
ros2 bag record /robot_force /robot/external_wrench /robot/ee_pose -o $BAG_NAME &
BAG_PID=$!

# Wait until bag folder exists (recorder is ready)
echo "Waiting for bag to initialize..."
until [ -d "${BAG_NAME}" ]; do
    sleep 0.05
done
echo "Bag confirmed recording, waiting ${DELAY}s before sending force command..."

sleep $DELAY

# Send commanded external force (effective_force = 0.5 * robot_force_)
ros2 topic pub --once /robot_force geometry_msgs/msg/Vector3 "{x: $FX, y: $FY, z: $FZ}"

echo "Force sent. Press Ctrl+C when motion is complete to stop recording."
echo "Commanded force = /robot_force, measured = /robot/external_wrench from the bag."
wait $BAG_PID
