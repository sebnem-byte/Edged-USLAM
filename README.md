# Edged-USLAM: Event-based Visual Inertial Odometry

This repository contains the **Edged-USLAM** workspace, a modified and containerized version of the visual-inertial odometry system designed for event cameras (based on RPG Ultimate SLAM).

## 🐳 Installation (Docker - Highly Recommended)

This project relies on specific dependencies (older ROS versions, specific Eigen/Ceres libraries). To avoid dependency hell, **we strongly recommend using the pre-built Docker container.**

### 1. Pull the Image
You can pull the ready-to-use image directly from GitHub Container Registry:

    docker pull ghcr.io/sebnem-byte/ze_vio_container:v1

### 2. Run the Container
To run the container with GUI support (rviz coverage), use the following command:

    xhost +local:docker
    docker run -it --net=host --privileged \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        ghcr.io/sebnem-byte/ze_vio_container:v1 /bin/bash

---

## 🛠 Manual Installation (Advanced)

If you insist on building it natively, ensure your environment meets the following strict requirements:

* **OS:** Ubuntu 16.04 (Xenial) or 18.04 (Bionic)
* **ROS:** Kinetic or Melodic
* **Build System:** `catkin_tools`
* **Dependencies:**
    * `eigen_catkin`
    * `ceres_catkin`
    * `glog_catkin`, `gflags_catkin`
    * `minkindr`, `rpg_dvs_ros`
    * OpenCV 3.x

*Note: You will need to manually resolve dependency conflicts if you are not using the Docker container.*

---

## ⚙️ Configuration & Calibration

Before running the system, you must ensure the calibration files match your sensor setup.

### 1. Camera & IMU Calibration (YAML)
The calibration files are located in:
`applications/ze_vio_ceres/data/`

You can create or edit a `.yaml` file (e.g., `DAVIS-example.yaml`) to define:
* **Intrinsics:** Camera matrix (fx, fy, cx, cy) and distortion coefficients.
* **Extrinsics (T_B_C):** Transformation matrix between the IMU (Body) and the Camera.
* **Resolution:** Image width and height.

### 2. Parameter Tuning (.cfg)
To fine-tune the algorithm or change ROS topic names, navigate to the `cfg/` folder inside `ze_vio_ceres`. Here you can modify:
* **Topic Names:** If your bag file uses different topics (e.g., `/dvs/events` vs `/cam0/events`).
* **Ceres Settings:** Optimization iterations, sliding window size, and marginalization settings.
* **Feature Tracking:** Keyframe selection criteria and feature detection thresholds.

---

## 🚀 Usage

### 1. Online / Live Mode (with Rosbag)
This mode simulates a live sensor stream using a rosbag.

**Step 1: Start RO Core**

    roscore

**Step 2: Play the Dataset**

    rosbag play path/to/your_data.bag

**Step 3: Launch the VIO Node**
Use the `live_` launch file for online processing. You may need to adjust the `timeshift` parameter to synchronize the IMU and Camera timestamps manually if they are not hardware-synced.

    roslaunch ze_vio_ceres live_DAVIS240C.launch \
        camera_name:=DAVIS-example \
        timeshift_cam_imu:=0.0028100209382249794

* `camera_name`: Must match the filename in the `data/` folder (e.g., `DAVIS-example.yaml`).
* `timeshift_cam_imu`: Time offset between camera and IMU (in seconds).

### 2. Offline Mode (Batch Processing)
For processing a bag file as fast as possible (non-real-time), use the standard launch files (without the `live_` prefix).

    roslaunch ze_vio_ceres davis240c.launch \
        dataset:=/path/to/your_data.bag \
        camera_name:=DAVIS-example

---

## 📊 Visualization
The launch files are configured to automatically open **RViz**. You can view:
* Estimated Trajectory (Path)
* Sparse 3D Map (Point Cloud)
* Current Frame & Feature Tracks

---

### Author
**Sebnem Byte**
*Docker Container & Workspace modifications.*
