# Edged-USLAM: Event-based Visual Inertial Odometry

This repository contains the **Edged-USLAM** workspace, a modified and containerized version of the visual-inertial odometry system designed for event cameras (based on RPG Ultimate SLAM).

## 🐳 Installation (Docker - Highly Recommended)

To avoid dependency conflicts and ensure reproducibility, **we strongly recommend using the pre-built Docker container.**

### 1. Pull the Image
You can pull the ready-to-use image directly from GitHub Container Registry:

    docker pull ghcr.io/sebnem-byte/ze_vio_container:v1

### 2. Run the Container
To run the container with GUI support (RViz etc.), use the following command:

    xhost +local:docker
    docker run -it --net=host --privileged \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        ghcr.io/sebnem-byte/ze_vio_container:v1 /bin/bash

---

## 🛠 Manual Installation (Native)

If you prefer to build the workspace natively on your host machine, ensure you are using the correct ROS version.

### System Requirements
* **OS:** Ubuntu 20.04 (Focal Fossa)
* **ROS Version:** ROS Noetic
* **Build System:** `catkin_tools` (Python 3)

### Dependencies
Ensure you have the standard build tools and ROS dependencies installed:

    sudo apt-get install python3-catkin-tools python3-vcstool
    sudo apt-get install ros-noetic-desktop-full

*Note: This project uses custom versions of `catkin_simple`, `eigen_catkin`, and `ceres_catkin` included in this workspace. Do not install conflicting system versions of these libraries.*

---

## ⚙️ Configuration & Calibration

Before running the system, ensure the calibration files match your sensor setup.

### 1. Camera & IMU Calibration (YAML)
Location: `applications/ze_vio_ceres/data/`

Edit or create a `.yaml` file (e.g., `DAVIS-example.yaml`) to define:
* **Intrinsics:** Camera matrix and distortion coefficients.
* **Extrinsics (T_B_C):** Transformation between IMU and Camera.

### 2. Parameter Tuning (.cfg)
Location: `applications/ze_vio_ceres/cfg/`

You can modify ROS topic names and algorithm parameters here (e.g., feature tracking thresholds, optimization window size).

---

## 🚀 Usage

### 1. Online / Live Mode (with Rosbag)
This mode simulates a live sensor stream using a rosbag.

**Step 1: Start ROS Core**

    roscore

**Step 2: Play the Dataset**

    rosbag play path/to/your_data.bag

**Step 3: Launch the VIO Node**
Use the `live_` launch file. Adjust `timeshift` if your bag file timestamps are not synchronized.

    roslaunch ze_vio_ceres live_DAVIS240C.launch \
        camera_name:=DAVIS-example \
        timeshift_cam_imu:=0.0

* `camera_name`: Must match the filename in the `data/` folder.

### 2. Offline Mode (Batch Processing)
For processing a bag file as fast as possible (non-real-time):

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
*Docker Container & Workspace modifications for ROS Noetic.*
