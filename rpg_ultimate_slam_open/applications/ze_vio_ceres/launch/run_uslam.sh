#!/bin/bash

source /opt/ros/noetic/setup.bash
source /home/edged_uslam/devel/setup.bash
ROSNODE_PATH="/home/edged_uslam/src/Edged-USLAM/rpg_ultimate_slam_open/applications/ze_vio_ceres"
BAGPATH="${1:-$(find /home/bags -name '*.bag' | fzf)}"
BAGNAME=$(basename "$BAGPATH" .bag)
LOGPATH="/tmp/${BAGNAME}_edged_$(date '+%F_%H-%M-%S')/"
mkdir -p /tmp/uslamrun
roslaunch ze_vio_ceres dvx_micro_events_only.launch bag_filename:="${BAGPATH}" # INSERT YOUR LAUNCHFILE HERE
mkdir -p "$LOGPATH"
cp /tmp/uslamrun/* "$LOGPATH"/
# copy config and launch files to results folder
cp "$ROSNODE_PATH"/launch/dvx_micro_events_only.launch "$ROSNODE_PATH"/cfg/vio_base.conf "$ROSNODE_PATH"/cfg/vio_dvx.conf "$LOGPATH"/
chown erik:erik "$LOGPATH" # REPLACE WITH YOUR HOST USER
# copy traj and param files to were the bag lives
cp -r "$LOGPATH" $(dirname "$BAGPATH")/
