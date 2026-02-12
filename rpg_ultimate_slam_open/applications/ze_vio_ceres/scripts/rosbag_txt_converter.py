#!/usr/bin/env python3
import rosbag

bag_path = "/media/sebnem/T7/tez/test_results/results_02_05/results_circle1/orb_slam3.bag"  # <-- .bag dosya adını buraya yaz

topic_name = "/orb_slam3/camera_pose"


#output_path = "/media/sebnem/T7/tez/test_results/results_02_05/results_circle1/trajectory.txt"
output_path = "/media/sebnem/T7/tez/test_results/results_02_05/results_circle1/aligned_traj_orbslam3.txt"

with rosbag.Bag(bag_path, "r") as bag, open(output_path, "w") as f_out:
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        timestamp = msg.header.stamp.to_sec()

        # Position
        tx = msg.pose.pose.position.x
        ty = msg.pose.pose.position.y
        tz = msg.pose.pose.position.z

        # Orientation (quaternion)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Write to file
        f_out.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

print(f"TUM trajectory saved to: {output_path}")
