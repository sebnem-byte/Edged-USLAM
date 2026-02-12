#!/usr/bin/env python3
import rosbag

bag = rosbag.Bag("/home/sebnem/edged_uslam_ws/src/rpg_ultimate_slam_open/data/shapes_translation.bag")
with open("/home/sebnem/edged_uslam_ws/src/rpg_ultimate_slam_open/data/shapes_translation/groundtruth.txt", "w") as f:
    for topic, msg, t in bag.read_messages(topics=["/optitrack/davis"]):
        ts = msg.header.stamp.to_sec()
        p = msg.pose.position
        q = msg.pose.orientation
        f.write(f"{ts:.9f} {p.x:.6f} {p.y:.6f} {p.z:.6f} {q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n")
bag.close()
