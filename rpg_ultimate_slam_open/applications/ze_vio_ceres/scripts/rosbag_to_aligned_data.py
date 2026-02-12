#!/usr/bin/env python3
import rosbag
import numpy as np

# ---- INPUT ----
bag_path = "/media/sebnem/T7/tez/test_results/results_02_05/results_circle1/dark5.bag"
odom_topic = "/ze_vio/odometry"
pose_topic = "/orb_slam3/camera_pose"
gt_topic = "/local_pose_vicon/pose"
output_gt = "/media/sebnem/T7/tez/test_results/results_02_05/results_circle1/aligned_gt_dark5.txt"
#output_odom = "/media/sebnem/T7/tez/test_results/results_02_05/results_circle1/aligned_traj_real_uslam.txt"
output_odom = "/media/sebnem/T7/tez/test_results/results_02_05/results_circle1/aligned_traj_dark5.txt"
max_diff = 0.02  # saniye

# ---- READ BAG ----
gt_msgs = []
odom_msgs = []
pose_msgs = []

print("Reading bag file...")

with rosbag.Bag(bag_path, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[gt_topic, odom_topic, pose_topic]):
        timestamp = msg.header.stamp.to_sec()
        if topic == gt_topic:
            p = msg.pose.position
            q = msg.pose.orientation
            gt_msgs.append((timestamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        elif topic == odom_topic:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            odom_msgs.append((timestamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        elif topic == pose_topic:
            p = msg.pose.position
            q = msg.pose.orientation
            pose_msgs.append((timestamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

        

gt_array = np.array(gt_msgs)
odom_array = np.array(odom_msgs)
pose_array = np.array(pose_msgs)

print(f"Loaded {len(gt_array)} GT poses and {len(odom_array)} odometry poses.")

# ---- SYNC ----
aligned_gt, aligned_odom = [], []

for gt in gt_array:
    diffs = np.abs(odom_array[:, 0] - gt[0])
    min_idx = np.argmin(diffs)
    if diffs[min_idx] < max_diff:
        aligned_gt.append(gt)
        aligned_odom.append(odom_array[min_idx])

#for gt in gt_array:
#    diffs = np.abs(pose_array[:, 0] - gt[0])
#    min_idx = np.argmin(diffs)
#    if diffs[min_idx] < max_diff:
#        aligned_gt.append(gt)
#        aligned_odom.append(pose_array[min_idx])

aligned_gt = np.array(aligned_gt)
#aligned_odom = np.array(aligned_odom)
aligned_odom = np.array(aligned_odom)

# ---- SAVE ----
np.savetxt(output_gt, aligned_gt, fmt="%.9f")
np.savetxt(output_odom, aligned_odom, fmt="%.9f")

print(f"Saved {len(aligned_gt)} aligned pairs to {output_gt} and {output_odom}")
