// This code is released only for internal evaluation.
// No commercial use, editing or copying is allowed.

#include <ze/vio_frontend/frontend_base.hpp>

#include <algorithm>
#include <iostream>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <imp/bridge/opencv/image_cv.hpp>
#include <imp/correspondence/epipolar_matcher.hpp>
#include <imp/correspondence/klt.hpp>
#include <imp/features/keypoints_wrapper.hpp>
#include <imp/features/feature_detector.hpp>
#include <ze/cameras/camera_rig.hpp>
#include <ze/common/combinatorics.hpp>
#include <ze/common/logging.hpp>
#include <ze/common/file_utils.hpp>
#include <ze/common/statistics.hpp>
#include <ze/data_provider/data_provider_factory.hpp>
#include <ze/geometry/triangulation.hpp>
#include <ze/geometry/pose_optimizer.hpp>
#include <ze/vio_common/frame.hpp>
#include <ze/vio_common/imu_integrator.hpp>
#include <ze/vio_common/landmark_triangulation.hpp>
#include <ze/vio_common/landmark_utils.hpp>
#include <ze/vio_common/nframe.hpp>
#include <ze/vio_common/vio_visualizer.hpp>
#include <ze/vio_frontend/frontend_gflags.hpp>
#include <ze/vio_frontend/feature_tracker.hpp>
#include <ze/vio_frontend/feature_initializer.hpp>
#include <ze/vio_frontend/stereo_matcher.hpp>
#include <ze/vio_frontend/track_extractor.hpp>
#include <ze/visualization/viz_ros.hpp>
#include <ze/data_provider/camera_imu_synchronizer.hpp>
#include <ze/nlls/imu_error.hpp>

#include <sophus/se3.hpp>

#include <Eigen/Dense>
#include <opencv2/core.hpp>


DEFINE_bool(vio_activate_backend, true,
            "Use visual-inertial backend.");
DEFINE_int32(vio_add_every_nth_frame_to_backend, -1,
            "Add every n-th non-Keyframe to backend.");
DEFINE_double(vio_disparity_median_for_static_motion_classification, 1.75,
              "Max disparity such that motion is classified as static [px].");
DEFINE_double(vio_ransac_relpose_thresh_px, 2,
              "Relative pose RANSAC reprojection error threshold [px].");
DEFINE_bool(vio_use_5pt_ransac, false,
            "Use 5-point ransac. Alternative is 2-point RANSAC with gyro prior.");
DEFINE_double(vio_acc_bias_init_x, 0.0, "Acc bias [temporary].");
DEFINE_double(vio_acc_bias_init_y, 0.0, "Acc bias [temporary].");
DEFINE_double(vio_acc_bias_init_z, 0.0, "Acc bias [temporary].");
DEFINE_double(vio_gyr_bias_init_x, 0.0, "Gyro bias [temporary].");
DEFINE_double(vio_gyr_bias_init_y, 0.0, "Gyro bias [temporary].");
DEFINE_double(vio_gyr_bias_init_z, 0.0, "Gyro bias [temporary].");
DEFINE_bool(vio_log_performance, false,
            "Log additional performance statistics, e.g., number of tracked features.");

DEFINE_uint64(vio_max_landmarks, 1000,
              "Maximum number of landmarks we store in the map.");

DEFINE_int64(vio_frame_size, 15000, "Event frame size (number of events used to draw the event frame).");

DEFINE_bool(vio_do_motion_correction, true,
              "Motion correct the event images using the current camera motion and scene structure.");

DEFINE_int32(noise_event_rate, 20000,
             "Events per second regarded as noise.");

DEFINE_double(vio_frame_norm_factor, 3.0,
              "Normalization factor for event frames");

namespace ze {

// -----------------------------------------------------------------------------
FrontendBase::FrontendBase()
  : rig_(cameraRigFromGflags())
  , imu_integrator_(std::make_shared<ImuIntegrator>())
  , thread_pool_(rig_->size())
  , T_C_B_(rig_->T_C_B_vec())
{
  initModules();
  initDvs();
  VLOG(1) << "Initialized frontend with camera:\n" << *rig_;
}

// -----------------------------------------------------------------------------
FrontendBase::FrontendBase(const CameraRigPtr& rig)
  : rig_(rig)
  , imu_integrator_(std::make_shared<ImuIntegrator>())
  , thread_pool_(rig_->size())
  , T_C_B_(rig_->T_C_B_vec())
{
  initModules();
  initDvs();
  feature_initializer_->setMaskExistingFeatures(false);
  imu_integrator_->setBiases(
        Vector3(FLAGS_vio_acc_bias_init_x, FLAGS_vio_acc_bias_init_y, FLAGS_vio_acc_bias_init_z),
        Vector3(FLAGS_vio_gyr_bias_init_x, FLAGS_vio_gyr_bias_init_y, FLAGS_vio_gyr_bias_init_z));
  VLOG(1) << "Initialized frontend with camera:\n" << *rig_;

}

// -----------------------------------------------------------------------------
FrontendBase::~FrontendBase()
{}

cv::Mat& FrontendBase::getPrevFrame() {
  static cv::Mat prev_event_img;
  return prev_event_img;
}


cv::Mat& FrontendBase::getCurrentFrame() {
static cv::Mat event_img_;
return event_img_;
}

cv::Mat& FrontendBase::getPrevImg() {
  static cv::Mat prev_im;
  return prev_im;
}


cv::Mat& FrontendBase::getCurrentImg() {
  static cv::Mat curr_img;
  return curr_img;
  }

// -----------------------------------------------------------------------------
void FrontendBase::initModules()
{
  EpipolarMatcherOptions options;
  stereo_matcher_ =
      std::make_shared<StereoMatcher>(
        *rig_, FLAGS_imp_detector_border_margin, FLAGS_vio_ransac_relpose_thresh_px,
        landmarks_);
  feature_tracker_ =
      std::make_shared<FeatureTracker>(
        *rig_, FLAGS_imp_detector_border_margin, FLAGS_vio_ransac_relpose_thresh_px,
        FLAGS_vio_use_5pt_ransac, *stereo_matcher_, landmarks_);
  feature_initializer_ =
      std::make_shared<FeatureInitializer>(
        *rig_, FLAGS_imp_detector_border_margin, landmarks_);
  visualizer_ = std::make_shared<VioVisualizer>(
        std::make_shared<VisualizerRos>(), rig_->stereoPairs());

  if (FLAGS_vio_viz_sleep != 0)
  {
    visualizer_->startThread();
  }

  //! @todo: Input queue does not work together with
  //!        addImuData callback.
//  if (FLAGS_vio_use_input_queue)
//  {
//    startThread();
//  }

  timers_[Timer::wait_time].start();
}

void FrontendBase::initDvs() {
  Camera::Ptr dvs_cam;
  if (rig_->getDvsCamera(&dvs_cam)) {
    // Initialize dvs_img_ to black.
    dvs_img_ = cv::Mat::zeros(dvs_cam->height(), dvs_cam->width(), CV_8U);
    //prev_event_im = cv::Mat::zeros(dvs_cam->height(), dvs_cam->width(), CV_8U);  // Get the persistent frame
    //event_img_ = cv::Mat::zeros(dvs_cam->height(), dvs_cam->width(), CV_32F);  // Get the persistent frame'
  
  }
}

// -----------------------------------------------------------------------------
// Add image data to queue
void FrontendBase::addData(
    const std::vector<std::pair<int64_t, ImageBase::Ptr>>& stamped_images,
    const std::vector<ImuStamps>& imu_stamps_vec,
    const std::vector<ImuAccGyrContainer>& imu_accgyr_vec)
{
  //  if (FLAGS_vio_use_input_queue)
  if (false)
  {
    FrontendInputData data;
    data.stamped_images = stamped_images;
    data.imu_stamps_vec = imu_stamps_vec;
    data.imu_accgyr_vec = imu_accgyr_vec;
    //! @todo(cfo): this can't be non-blocking since we need to have all IMU
    //!             measurements! implement a fast processData function that
    //!             skips images but no IMU measurements.
    input_queue_.write(data);
  }
  else
  {
    processData(stamped_images, imu_stamps_vec, imu_accgyr_vec);
  }
}

// Add events data to queue
void FrontendBase::addData(
    const std::pair<int64_t, EventArrayPtr>& stamped_events,
    const std::vector<ImuStamps>& imu_stamps_vec,
    const std::vector<ImuAccGyrContainer>& imu_accgyr_vec)
{
  //  if (FLAGS_vio_use_input_queue)
  if (false)
  {
    FrontendInputData data;
    data.stamped_events = stamped_events;
    data.imu_stamps_vec = imu_stamps_vec;
    data.imu_accgyr_vec = imu_accgyr_vec;
    //! @todo(cfo): this can't be non-blocking since we need to have all IMU
    //!             measurements! implement a fast processData function that
    //!             skips images but no IMU measurements.
    input_queue_.write(data);
  }
  else
  {
    processData(stamped_events, imu_stamps_vec, imu_accgyr_vec);
  }
}



real_t FrontendBase::estimateDepthFromOpticalFlow(const cv::Mat& prev_frame,
                                                  const cv::Mat& curr_frame,
                                                  double fx,
                                                  double baseline_meters)
{

  // ⬇️ Downsample (örnek: 2 kat küçült)
  cv::Mat prev_resized, curr_resized;
  cv::resize(prev_frame, prev_resized, cv::Size(), 0.5, 0.5);
  cv::resize(curr_frame, curr_resized, cv::Size(), 0.5, 0.5);

  fx *= 0.5; // resize oranına göre

  // Optical Flow hesapla
  cv::Mat flow;
  cv::calcOpticalFlowFarneback(prev_resized, curr_resized, flow,0.5, 1, 7, 3, 2, 1.1, 0);
  // Flow bileşenlerini ayır
  std::vector<cv::Mat> flow_channels(2);
  cv::split(flow, flow_channels);
  cv::Mat magnitude;
  cv::magnitude(flow_channels[0], flow_channels[1], magnitude);

  // Ortalama displacement
  cv::Scalar mean_disp = cv::mean(magnitude);
  double avg_pixel_shift = mean_disp[0];

  if (avg_pixel_shift < 1e-3 || std::isnan(avg_pixel_shift))
    return -1.0;

  // Derinlik (Z) hesabı
  real_t depth = (baseline_meters * fx) / (avg_pixel_shift + 1e-5);
  return static_cast<real_t>(depth*0.05);

}


real_t FrontendBase::computeMedianDepthFromLandmarks()
{
  std::vector<real_t> depths;

  const size_t cam_idx = 0;
  const ze::NFrame::Ptr nframe = states_.nframeKm1();

  if (!nframe || nframe->size() <= cam_idx) {
    LOG(WARNING) << "nframeKm1 is null or missing camera index.";
    return -1.0;
  }

  const ze::Frame& frame = nframe->at(cam_idx);
  const Transformation T_C_W = T_C_B_.at(0) * states_.T_Bkm1_W().inverse();


  for (uint32_t slot = 0; slot < landmarks_.numLandmarks(); ++slot)
  {
    if (!landmarks_.isSlotActive(slot)) {
      continue;
    }

    const Eigen::Vector3d& p_W = landmarks_.p_W_atSlot(slot);
    const Eigen::Vector3d p_C = T_C_W * p_W;

    if (p_C.z() <= 0.0) {
      continue;  // Kamera arkasındaki noktayı atla
    }
    depths.push_back(p_C.z());
  }


  if (depths.empty()) {
    //LOG(WARNING) << "No valid landmark depths.";
    return -1.0;
  }

  std::sort(depths.begin(), depths.end());
  return depths[depths.size() / 2];
}




// Add image and events data to queue
void FrontendBase::addData(
    const std::vector<std::pair<int64_t, ImageBase::Ptr>>& stamped_images,
    const std::pair<int64_t, EventArrayPtr>& stamped_events,
    const std::vector<ImuStamps>& imu_stamps_vec,
    const std::vector<ImuAccGyrContainer>& imu_accgyr_vec,
    const bool& no_motion_prior)
{
  processData(stamped_images, stamped_events, imu_stamps_vec, imu_accgyr_vec,
              no_motion_prior);
}

// -----------------------------------------------------------------------------
// Process Data coming from images
void FrontendBase::processData(
    const std::vector<std::pair<int64_t, ImageBase::Ptr>>& stamped_images,
    const std::vector<ImuStamps>& imu_stamps_vec,
    const std::vector<ImuAccGyrContainer>& imu_accgyr_vec)
{
  timers_[Timer::wait_time].stop();
  auto t_tot = timers_[Timer::total_time].timeScope();

#ifdef ZE_VIO_LIMITED
  if (limitNumberOfFrames(5000))
  {
    return;
  }
#endif

  motion_type_ = VioMotionType::NotComputed;

  // Add new imu messages to members imu_*_since_lkf_
  CHECK(!imu_stamps_vec.empty()) << "There are no IMU stamps";
  CHECK(!imu_accgyr_vec.empty()) << "There are no IMU measurements";
  if (!addImuMeasurementsBetweenKeyframes(imu_stamps_vec.at(0),
                                          imu_accgyr_vec.at(0))
      && FLAGS_num_imus > 0)
  {
    LOG(ERROR) << "Insufficient IMU messages provided.";
    return;
  }
  cleanupInactiveLandmarksFromLastIteration();

  if (states_.nframeKm1())
  {
    if (FLAGS_vio_activate_backend && update_states_cb_)
    {
      {
        // If optimizer has new result, this callback will copy them in our states.
        auto t = timers_[Timer::wait_for_backend].timeScope();
        // This is also integrating the IMU... and adding it to the odom member
        pollBackend(true);
      }


      {
        auto t = timers_[Timer::visualization].timeScope();
        visualizer_->copyDataAndVisualizeConcurrently(states_, landmarks_,
                                                      localization_information_);
      }

      {
        // Retriangulate all landmarks if we have new states.
        auto t = timers_[Timer::retriangulate_all_landmarks].timeScope();
        retriangulateAllLandmarks(states_, T_C_B_, landmarks_);
      }
    }

    // Update scene depth
    {
      real_t depth = states_.nframeKm1()->at(0).median_depth_;
      
      if (depth > FLAGS_vio_min_depth &&
          depth != FLAGS_vio_median_depth)
      {

        scene_depth_ = depth;
        VLOG(1) << "Scene depth: " << scene_depth_;
      }

    }
  }

  // Predict pose of new frame using IMU:
  Transformation T_Bkm1_Bk;
  // Integrate IMU to current image time stamp
  if (states_.nframeKm1())
  {
    if (!imu_stamps_vec.empty() && imu_stamps_vec.at(0).size() > 0)
    {
      Vector3 v_W = states_.v_W(states_.nframeHandleK());

      T_Bkm1_Bk = imu_integrator_->integrateImu(
            imu_stamps_vec.at(0),
            imu_accgyr_vec.at(0),
            states_.T_Bk_W(),
            v_W);
    }
  }

  // Create new frame:
  NFrame::Ptr nframe_k = createNFrame(stamped_images);
  VLOG(3) << " =============== Frame " << nframe_k->seq() << " - "
          << nframe_k->handle() << " ===============";

  // Predict pose of current frame by integrating the gyroscope:
  if (states_.nframeKm1())
  {
    states_.T_Bk_W() = T_Bkm1_Bk.inverse() * states_.T_Bkm1_W();
  }
  else
  {
    LOG(WARNING) << "Initialize T_Bk_W to identity.";
    states_.T_Bk_W() = Transformation();
  }

  //
  // Run tracking in derived class.
  //
  this->processData(T_Bkm1_Bk);

  // Process callbacks.
  if (tracked_nframe_cb_
      && (nframe_k->isKeyframe()
          || (FLAGS_vio_add_every_nth_frame_to_backend > 0
              && nframe_k->seq() % FLAGS_vio_add_every_nth_frame_to_backend == 0)))
  {
    std::vector<LandmarkHandle> lm_opportunistic;
    std::vector<LandmarkHandle> lm_persistent_new;
    std::vector<LandmarkHandle> lm_persistent_continued;
    if (stage_ == FrontendStage::Running && nframe_k->isKeyframe())
    {
      // Extract feature tracks to be processed by backend.
      auto t = timers_[Timer::track_selection].timeScope();
      DEBUG_CHECK(motion_type_ != VioMotionType::NotComputed);
      selectLandmarksForBackend(
            motion_type_, T_C_B_, states_, *nframe_k, landmarks_,
            lm_opportunistic, lm_persistent_new, lm_persistent_continued);
    }

    {
      // Extract feature descriptors.
      auto t = timers_[Timer::descriptor_extraction].timeScope();
      feature_initializer_->extractFeatureDescriptors(*nframe_k);
    }

    if (FLAGS_vio_activate_backend)
    {
      // Run optimization.
      auto t = timers_[Timer::add_frame_to_backend].timeScope();
      tracked_nframe_cb_(
            nframe_k, imu_stamps_since_lkf_, imu_accgyr_since_lkf_, motion_type_,
            lm_opportunistic, lm_persistent_new, lm_persistent_continued);
    }
  }

  if (nframe_k->isKeyframe())
  {
    states_.setKeyframe(nframe_k->handle());
    imu_stamps_since_lkf_.resize(0);
    imu_accgyr_since_lkf_.resize(6,0);
  }

  if (result_cb_)
  {
    const Transformation T_W_B = states_.T_Bk_W().inverse();
    result_cb_(
          nframe_k->timestamp(),
          T_W_B.getEigenQuaternion().cast<double>(),
          T_W_B.getPosition().cast<double>(),
          stage_,
          0u //! @todo: return num tracked keypoints!
          );
  }

  if (FLAGS_vio_log_performance)
  {
    logNumTrackedFeatures(*nframe_k, landmarks_);
  }
  timers_[Timer::wait_time].start();
}


// Process Data coming from events
void FrontendBase::processData(
    const std::pair<int64_t, EventArrayPtr>& stamped_events,
    const std::vector<ImuStamps>& imu_stamps_vec,
    const std::vector<ImuAccGyrContainer>& imu_accgyr_vec)
{

  timers_[Timer::wait_time].stop();
  auto t_tot = timers_[Timer::total_time].timeScope();

#ifdef ZE_VIO_LIMITED
  if (limitNumberOfFrames(5000))
  {
    return;
  }
#endif

  motion_type_ = VioMotionType::NotComputed;
  CHECK(!imu_stamps_vec.empty()) << "There are no IMU stamps";
  CHECK(!imu_accgyr_vec.empty()) << "There are no IMU measurements";
  if (!addImuMeasurementsBetweenKeyframes(imu_stamps_vec.at(0),
                                          imu_accgyr_vec.at(0))
      && FLAGS_num_imus > 0)
  {
    LOG(ERROR) << "No IMU messages provided.";
    return;
  }
  cleanupInactiveLandmarksFromLastIteration();

  // Update state from backend
  if (states_.nframeKm1() &&
      FLAGS_vio_activate_backend && update_states_cb_)
  {
    {
      // If optimizer has new result, this callback will copy them in our states.
      auto t = timers_[Timer::wait_for_backend].timeScope();
      pollBackend(true);
    }


    {
      auto t = timers_[Timer::visualization].timeScope();
      visualizer_->copyDataAndVisualizeConcurrently(states_, landmarks_,
                                                    localization_information_);
    }

    {
      // Retriangulate all landmarks if we have new states.
      auto t = timers_[Timer::retriangulate_all_landmarks].timeScope();
      retriangulateAllLandmarks(states_, T_C_B_, landmarks_);
    }

    // Update scene depth
    {
      scene_depth_ = states_.nframeKm1()->at(0).median_depth_;
      //real_t k_depth = static_cast<real_t>(0.96);
      //real_t depth = states_.nframeKm1()->at(0).median_depth_*k_depth;   
      //if (depth>5 || depth<0)
      //{
      //  depth = 1*k_depth;
      //} else
      //{
      //  depth= depth;
      //}
      //scene_depth_= depth;
      VLOG(1) << "Scene depth: " << scene_depth_;
    }
  }

  // Predict pose of new frame using IMU:
  Transformation T_Bkm1_Bk;
  if (states_.nframeKm1())
  {
    if (!imu_stamps_vec.empty() && imu_stamps_vec.at(0).size() > 0)
    {
      Vector3 v_W = states_.v_W(states_.nframeHandleK());

      T_Bkm1_Bk = imu_integrator_->integrateImu(
            imu_stamps_vec.at(0),
            imu_accgyr_vec.at(0),
            states_.T_Bk_W(),
            v_W);
     }
  }

  //---------------------------------------------------------------------//
  // Events related things

  CHECK(rig_->dvs_bearing_lut_.size() > 0) << "Bearing lookup table is empty";

  int64_t t0 = imu_stamps_vec.at(0).head(1)[0];
  int64_t t1 = imu_stamps_vec.at(0).tail(1)[0];

  // Create new stamped_images from stamped_events
  cv::Mat event_img = cv::Mat::zeros(rig_->at(0).height(), rig_->at(0).width(), CV_32F);
  const EventArrayPtr events_ptr = stamped_events.second;
  VLOG(20) << "Number of events: " << events_ptr->size();

  size_t n_events_for_noise_detection = std::min(events_ptr->size(), size_t(4000));
  // Event_rate = (# of events) / (time difference between last and first of these events)
  // Refactor this calculation, what if the denominator is 0, negative, etc...
  // What happens when you get only one event... Then denom is 0..
  real_t event_rate = n_events_for_noise_detection /
         (events_ptr->back().ts -
          events_ptr->at(events_ptr->size()-n_events_for_noise_detection).ts).toSec();

  if (event_rate < FLAGS_noise_event_rate)
  {
    motion_type_ = VioMotionType::NoMotion;
  }
  else
  {
    auto t = timers_[Timer::draw_events].timeScope();

    // Build event frame with fixed number of events
    const size_t winsize_events = FLAGS_vio_frame_size;
    VLOG(10) << "Window size: " << winsize_events << " events";

    int first_idx = std::max((int)events_ptr->size() - (int) winsize_events, 0);

    uint64_t frame_length =
        events_ptr->back().ts.toNSec() - events_ptr->at(first_idx).ts.toNSec();
    stats_[Stats::frame_length].addSample(frame_length);

    if(events_ptr->size() < winsize_events)
    {
      VLOG(1) << "Requested frame size of length " << winsize_events
                   << " events, but I only have "
                   << events_ptr->size()
                   << " events in the last event array";
    }

    Transformation T_1_0 = T_C_B_[0] * T_Bkm1_Bk.inverse() * T_C_B_[0].inverse();

    drawEvents(
      events_ptr->begin()+first_idx,
      events_ptr->end(),
      t0, t1,
      T_1_0,
      event_img);

    const float bmax = FLAGS_vio_frame_norm_factor;
    event_img.convertTo(event_img, CV_8U, 255./bmax);
    //cv::normalize(event_img, event_img, 0, 255, cv::NORM_MINMAX, CV_8U);

    dvs_img_ = event_img;
  }


  std::shared_ptr<ze::ImageCv8uC1> img_cv_ptr
      = std::make_shared<ze::ImageCv8uC1>(dvs_img_, ze::PixelOrder::gray);
  StampedImages stamped_images;

  const int64_t stamp = stamped_events.first;

  stamped_images.push_back(std::pair<int64_t, ImageBase::Ptr>(stamp, img_cv_ptr));
  //---------------------------------------------------------------------//

  // Create new frame:
  NFrame::Ptr nframe_k = createNFrame(stamped_images);
  VLOG(3) << " =============== Frame " << nframe_k->seq() << " - "
          << nframe_k->handle() << " ===============";

  // Predict pose of current frame by integrating the gyroscope:
  if (states_.nframeKm1())
  {
    states_.T_Bk_W() = T_Bkm1_Bk.inverse() * states_.T_Bkm1_W();
  }
  else
  {
    LOG(WARNING) << "Initialize T_Bk_W to identity.";
    states_.T_Bk_W() = Transformation();
  }

  //
  // Run tracking in derived class.
  //
  this->processData(T_Bkm1_Bk);

  // Process callbacks.
  if (tracked_nframe_cb_
      && (nframe_k->isKeyframe()
          || (FLAGS_vio_add_every_nth_frame_to_backend > 0
              && nframe_k->seq() % FLAGS_vio_add_every_nth_frame_to_backend == 0)))
  {
    std::vector<LandmarkHandle> lm_opportunistic;
    std::vector<LandmarkHandle> lm_persistent_new;
    std::vector<LandmarkHandle> lm_persistent_continued;
    if (stage_ == FrontendStage::Running && nframe_k->isKeyframe())
    {
      // Extract feature tracks to be processed by backend.
      auto t = timers_[Timer::track_selection].timeScope();
      DEBUG_CHECK(motion_type_ != VioMotionType::NotComputed);
      selectLandmarksForBackend(
            motion_type_, T_C_B_, states_, *nframe_k, landmarks_,
            lm_opportunistic, lm_persistent_new, lm_persistent_continued);
    }

    {
      // Extract feature descriptors.
      auto t = timers_[Timer::descriptor_extraction].timeScope();
      feature_initializer_->extractFeatureDescriptors(*nframe_k);
    }

    if (FLAGS_vio_activate_backend)
    {
      // Run optimization.
      auto t = timers_[Timer::add_frame_to_backend].timeScope();
      tracked_nframe_cb_(
            nframe_k, imu_stamps_since_lkf_, imu_accgyr_since_lkf_, motion_type_,
            lm_opportunistic, lm_persistent_new, lm_persistent_continued);
    }
  }

  if (nframe_k->isKeyframe())
  {
    states_.setKeyframe(nframe_k->handle());
    imu_stamps_since_lkf_.resize(0);
    imu_accgyr_since_lkf_.resize(6,0);
  }

  if (result_cb_)
  {
    const Transformation T_W_B = states_.T_Bk_W().inverse();
    result_cb_(
          nframe_k->timestamp(),
          T_W_B.getEigenQuaternion().cast<double>(),
          T_W_B.getPosition().cast<double>(),
          stage_,
          0u //! @todo: return num tracked keypoints!
          );
  }

  if (FLAGS_vio_log_performance)
  {
    logNumTrackedFeatures(*nframe_k, landmarks_);
  }
  timers_[Timer::wait_time].start();
}

// Process Data coming from images and events
void FrontendBase::processData(
    const std::vector<std::pair<int64_t, ImageBase::Ptr>>& stamped_images,
    const std::pair<int64_t, EventArrayPtr>& stamped_events,
    const std::vector<ImuStamps>& imu_stamps_vec,
    const std::vector<ImuAccGyrContainer>& imu_accgyr_vec,
    const bool& no_motion_prior)
{
  CHECK(!stamped_images.empty());
  CHECK_NOTNULL(stamped_images.at(0).second.get());

  timers_[Timer::wait_time].stop();
  auto t_tot = timers_[Timer::total_time].timeScope();

#ifdef ZE_VIO_LIMITED
  if (limitNumberOfFrames(5000))
  {
    return;
  }
#endif

  motion_type_ = VioMotionType::NotComputed;
  CHECK(!imu_stamps_vec.empty()) << "There are no IMU stamps";
  CHECK(!imu_accgyr_vec.empty()) << "There are no IMU measurements";
  if (!addImuMeasurementsBetweenKeyframes(imu_stamps_vec.at(0),
                                          imu_accgyr_vec.at(0))
      && FLAGS_num_imus > 0)
  {
    LOG(ERROR) << "No IMU messages provided.";
    return;
  }
  cleanupInactiveLandmarksFromLastIteration();

  // Update state from backend
  if (states_.nframeKm1() &&
      FLAGS_vio_activate_backend && update_states_cb_)
  {
    {
      // If optimizer has new result, this callback will copy them in our states.
      auto t = timers_[Timer::wait_for_backend].timeScope();
      pollBackend(true);
    }


    {
      auto t = timers_[Timer::visualization].timeScope();
      visualizer_->copyDataAndVisualizeConcurrently(states_, landmarks_,
                                                    localization_information_);
    }

    {
      // Retriangulate all landmarks if we have new states.
      auto t = timers_[Timer::retriangulate_all_landmarks].timeScope();
      retriangulateAllLandmarks(states_, T_C_B_, landmarks_);
    }

    // Update scene depth
    {
        //scene_depth_ = states_.nframeKm1()->at(0).median_depth_;
        real_t depth;
        // Depth estimation with optical flow 
           // Persistent previous and current event frames
        /*                  
         cv::Mat& prev_im = getPrevImg();  // Get the persistent frame
         cv::Mat& curr_img = getCurrentImg();  // Get the persistent frame
   
         dvs_img_.copyTo(curr_img);
         const double fx = static_cast<double>(199.37143467278824);
   
         if (prev_im.empty()) {
           prev_im= cv::Mat::zeros(curr_img.size(), CV_8U); 
         }
           // Kamera parametreleri (örnek)
           Camera::ConstPtr dvs_cam;
           if (!rig_->getDvsCamera(&dvs_cam)) {
           const VectorX& params = dvs_cam->projectionParameters();
           const double fx = static_cast<double>(params[0]);
           const double fy = static_cast<double>(params[1]);
           const double cx = static_cast<double>(params[2]);
           const double cy = static_cast<double>(params[3]);}
           
           const double baseline = 0.1; // metre cinsinden keyframe arası mesafe
           
           // Derinlik tahmini
           if (!curr_img.empty() || !prev_im.empty())
           {
            curr_img.convertTo(curr_img,CV_8U);
            prev_im.convertTo(prev_im,CV_8U);
            std::cout << "prev_gray.empty(): " << curr_img.empty() << std::endl;
            std::cout << "curr_gray.empty(): " << prev_im.empty() << std::endl;
            depth = estimateDepthFromOpticalFlow(prev_im, curr_img, fx, baseline); }
   
           // Assign the old frame as the prev event image
           if (prev_im.empty() || cv::countNonZero(prev_im) == 0 || curr_img.size() != prev_im.size()) {
             prev_im = cv::Mat::zeros(curr_img.size(), CV_8U); // Siyah bir frame ile başlat
           }
           else {
             curr_img.copyTo(prev_im);
   
            }*/

        // DEPTH estimation for real-time application    

        bool Real_Time_app = 1; 


        if (Real_Time_app)
        {
          real_t k_depth = static_cast<real_t>(0.99);
          depth = states_.nframeKm1()->at(0).median_depth_*k_depth;   
          if (depth>4 || depth<0)
          {
            depth = 0.5;
          } else
          {
            depth= depth*k_depth;
          }
      }
      else
      {
        // depth for online applications
       real_t k_depth = static_cast<real_t>(0.96);
       depth = computeMedianDepthFromLandmarks();   
       if (depth>3 || depth<0)
       {
          depth = states_.nframeKm1()->at(0).median_depth_;
          if (depth>3 || depth<0)
          {
            depth = 1;
          } else
          {
            depth= depth*k_depth;
          }
       } 
       else
       {
           depth = depth*k_depth;
       }
      }
      depth = states_.nframeKm1()->at(0).median_depth_;
      
      // scene_depth_ = depth;
     //real_t k_depth = static_cast<real_t>(0.95);
     //depth = computeMedianDepthFromLandmarks();   
     //if (depth>5 || depth<0)
     //{
     //   depth = states_.nframeKm1()->at(0).median_depth_;
     //} else
     //{
     //  //depth= states_.nframeKm1()->at(0).median_depth_;
     //  if (depth>5 || depth<0)
     //  {
     //    depth = depth*k_depth;
     //  }
     //}
      scene_depth_ = depth;
     //std::cout << "Scene depth: " << scene_depth_ << std::endl;

     //VLOG(1) << "Scene depth: " << scene_depth_;
    }
  }

  // Predict pose of new frame using IMU:
  Transformation T_Bkm1_Bk;
  if (states_.nframeKm1())
  {
    if (!imu_stamps_vec.empty() && imu_stamps_vec.at(0).size() > 0)
    {
      Vector3 v_W = states_.v_W(states_.nframeHandleK());

      T_Bkm1_Bk = imu_integrator_->integrateImu(
            imu_stamps_vec.at(0),
            imu_accgyr_vec.at(0),
            states_.T_Bk_W(),
            v_W);
     }
  }

  //---------------------------------------------------------------------//
  // Events related things

  /// Stamp of event frame.
  const int64_t& event_frame_timestamp = stamped_images.at(0).first;
  /// Generate frame out of events using the IMU.
  CHECK_NOTNULL(stamped_events.second.get());
  CHECK(rig_->dvs_bearing_lut_.size() > 0) << "Bearing lookup table is empty";

  const int64_t& t0 = imu_stamps_vec.at(0).head(1)[0];
  const int64_t& t1 = imu_stamps_vec.at(0).tail(1)[0];

  /// Initialize values
  /// Image is black.
  cv::Mat event_img = cv::Mat::zeros(
        rig_->at(0).height(), rig_->at(0).width(), CV_32F);

  /// If there are no events just send the previous image to the backend...
  /// The tracker sends nothing if we feed with black images.
  if (!stamped_events.second->empty()) {
    const EventArrayPtr& events_ptr = stamped_events.second;
    VLOG(20) << "Number of events: " << events_ptr->size();
    size_t n_events_for_noise_detection = std::min(events_ptr->size(), size_t(2000));
    // The event rate is only meaningful when creating the events package
    // using the last x events! If you take the events in a ceratain fixed time
    // frame (as was done before), then what happens if there are no events...
    real_t event_rate = n_events_for_noise_detection /
        (events_ptr->back().ts -
          events_ptr->at(events_ptr->size()-n_events_for_noise_detection).ts).toSec();


    // Only draw a new event image if the rate of events is sufficiently high
    // If not, then just use the previous drawn image in the backend.
    // This ensures that feature tracks are still tracked, and also induces
    // a subtile NoMotion prior to the backend: instead of directly using
    // NoMotion.
    if (event_rate >= FLAGS_noise_event_rate)
    {
      auto t = timers_[Timer::draw_events].timeScope();

      // Build event frame with fixed number of events
      const size_t winsize_events = FLAGS_vio_frame_size;
      VLOG(10) << "Window size: " << winsize_events << " events";

      int first_idx = std::max((int)events_ptr->size() - (int) winsize_events, 0);

      uint64_t frame_length =
          events_ptr->back().ts.toNSec() - events_ptr->at(first_idx).ts.toNSec();
      stats_[Stats::frame_length].addSample(frame_length);

      if(events_ptr->size() < winsize_events)
      {
        VLOG(1) << "Requested frame size of length " << winsize_events
                      << " events, but I only have "
                      << events_ptr->size()
                      << " events in the last event array";
      }

      /// Get Dvs transform from body to camera
      size_t dvs_camera_index;
      CHECK(rig_->getDvsCameraIndex(&dvs_camera_index));

      /// Express transformation from camera at time k minus 1 to
      /// camera at time k.
      Transformation T_1_0 = T_C_B_[dvs_camera_index] * T_Bkm1_Bk.inverse()
          * T_C_B_[dvs_camera_index].inverse();

      drawEvents(
            events_ptr->begin()+first_idx,
            events_ptr->end(),
            t0, t1,
            T_1_0,
            event_img);

      const float bmax = FLAGS_vio_frame_norm_factor;
      event_img.convertTo(event_img, CV_8U, 255./bmax);
      //cv::normalize(event_img, event_img, 0, 255, cv::NORM_MINMAX, CV_8U);

      dvs_img_ = event_img;
    } else {
      motion_type_ = VioMotionType::NoMotion;
    }
  } else {
      VLOG(1) << "There are no events.";
  }
  //---------------------------------------------------------------------//

  // Generate vector with the images to create an NFrame
  // Initialize from the normal images.
  StampedImages images_for_NFrame (stamped_images);

  // Add the generated image as if it was yet another image taken at the same
  // time than the normal image.
  /// Stamp as the normal image.
  images_for_NFrame.push_back(std::pair<int64_t, ImageBase::Ptr>(
                                event_frame_timestamp,
                                std::make_shared<ze::ImageCv8uC1>(dvs_img_,
                                                 ze::PixelOrder::gray)));

  // Create new frame:
  NFrame::Ptr nframe_k = createNFrame(images_for_NFrame);
  VLOG(3) << " =============== Frame " << nframe_k->seq() << " - "
          << nframe_k->handle() << " ===============";

  // Predict pose of current frame by integrating the gyroscope:
  if (states_.nframeKm1())
  {
    states_.T_Bk_W() = T_Bkm1_Bk.inverse() * states_.T_Bkm1_W();
  }
  else
  {
    LOG(WARNING) << "Initialize T_Bk_W to identity.";
    states_.T_Bk_W() = Transformation();
  }

  //
  // Run tracking in derived class.
  //
  this->processData(T_Bkm1_Bk);

  // Process callbacks.
  if (tracked_nframe_cb_
      && (nframe_k->isKeyframe()
          || (FLAGS_vio_add_every_nth_frame_to_backend > 0
              && nframe_k->seq() % FLAGS_vio_add_every_nth_frame_to_backend == 0)))
  {
    std::vector<LandmarkHandle> lm_opportunistic;
    std::vector<LandmarkHandle> lm_persistent_new;
    std::vector<LandmarkHandle> lm_persistent_continued;
    if (stage_ == FrontendStage::Running && nframe_k->isKeyframe())
    {
      // Extract feature tracks to be processed by backend.
      auto t = timers_[Timer::track_selection].timeScope();
      DEBUG_CHECK(motion_type_ != VioMotionType::NotComputed);
      selectLandmarksForBackend(
            motion_type_, T_C_B_, states_, *nframe_k, landmarks_,
            lm_opportunistic, lm_persistent_new, lm_persistent_continued);
    }

    {
      // Extract feature descriptors.
      auto t = timers_[Timer::descriptor_extraction].timeScope();
      feature_initializer_->extractFeatureDescriptors(*nframe_k);
    }

    if (FLAGS_vio_activate_backend)
    {
      // Run optimization.
      auto t = timers_[Timer::add_frame_to_backend].timeScope();
      tracked_nframe_cb_(
            nframe_k, imu_stamps_since_lkf_, imu_accgyr_since_lkf_, motion_type_,
            lm_opportunistic, lm_persistent_new, lm_persistent_continued);
    }
  }

  if (nframe_k->isKeyframe())
  {
    states_.setKeyframe(nframe_k->handle());
    imu_stamps_since_lkf_.resize(0);
    imu_accgyr_since_lkf_.resize(6,0);
  }

  if (result_cb_)
  {
    const Transformation T_W_B = states_.T_Bk_W().inverse();
    result_cb_(
          nframe_k->timestamp(),
          T_W_B.getEigenQuaternion().cast<double>(),
          T_W_B.getPosition().cast<double>(),
          stage_,
          0u //! @todo: return num tracked keypoints!
          );
  }

  if (FLAGS_vio_log_performance)
  {
    logNumTrackedFeatures(*nframe_k, landmarks_);
  }
  timers_[Timer::wait_time].start();
}


 

/**/
// WITH EDGE DETECTION and Nonlinear MCP module
// --------------------------------------------- NEW CODE START ------------------------------------------------------------
void FrontendBase::drawEvents(
  const EventArray::iterator& first,
  const EventArray::iterator& last,
  const int64_t& t0,
  const int64_t& t1,
  const Transformation& T_1_0,
  cv::Mat &out)
{

  size_t n_events = 0;
  Eigen::Matrix<float, 2, Eigen::Dynamic> events;

  // Persistent previous and current event frames
  //cv::Mat& prev_event_im = getPrevFrame();  // Get the persistent frame
  //cv::Mat& event_img_ = getCurrentFrame();  // Get the persistent frame

  //cv::Mat event_img_;  // Get the persistent frame
  events.resize(2, last - first);
  // DVS kamerayı alın
  Camera::ConstPtr dvs_cam;
  if (!rig_->getDvsCamera(&dvs_cam)) {
      LOG(WARNING) << "Draw events is called but there is no DVS camera in rig";
      return;
  }

  // Kamera boyutlarını alın
  const int height = dvs_cam->height();
  const int width = dvs_cam->width();

  CHECK_EQ(out.rows, height);
  CHECK_EQ(out.cols, width);

  const VectorX& params = dvs_cam->projectionParameters();
  const float fx = params[0];
  const float fy = params[1];
  const float cx = params[2];
  const float cy = params[3];
  Eigen::Matrix4f K;
  K << fx, 0., cx, 0.,
       0., fy, cy, 0.,
       0., 0., 1., 0.,
       0., 0., 0., 1.;


  // Çıkış matrisinin boyutlarını kontrol edin ve gerektiğinde yeniden boyutlandırın
  if (out.empty() || out.rows != height || out.cols != width) {
     out = cv::Mat::zeros(height, width, CV_32F);
  }


  //Eigen::Matrix4f T = K * T_1_0.getTransformationMatrix().cast<float>() * K.inverse();

  float depth = scene_depth_;

  bool do_motion_correction = FLAGS_vio_do_motion_correction;
  float dt = 0;

  
  //float dt = 0;
 Eigen::Matrix3f R = T_1_0.getRotation().cast<float>().toImplementation().toRotationMatrix();
 Eigen::Vector3f t = T_1_0.getPosition().cast<float>();


  static std::atomic<bool> imu_sat; 
  static std::atomic<bool> gyro_sat; 
  //imu_sat = ze::nlls::ImuError::imu_saturated.load();
  //gyro_sat = ze::nlls::ImuError::gyro_saturated.load();


  imu_sat = false;
  gyro_sat = false;

  static Eigen::Matrix3f last_R = Eigen::Matrix3f::Identity();
  static Eigen::Vector3f last_t = Eigen::Vector3f::Zero();


  //instead of using linear interpolation use SE(3) monifold (realistic camera motion)
    Sophus::SE3 T_rel(R.template cast<double>(), t.template cast<double>());
    Sophus::SE3 T0;
    Eigen::Matrix<double, 6, 1> log_T = T_rel.log();
  std::vector<Eigen::Vector3f> precomputed_rays(width * height);

  for (auto e = first; e != last; ++e)
  {  // 1. kindr'dan rotation ve translation al (double)
      {
          dt = static_cast<float>(t1 - e->ts.toNSec()) / (t1 - t0);
          // add clamp 
          dt = std::min(std::max(dt, 0.f), 1.f);

      }

      // Original pixel coordinates
      Eigen::Vector2f px = rig_->dvs_keypoint_lut_.col(e->x + e->y * width);

      // Reproject into normalized camera coordinates
      //Eigen::Vector3f ray((px[0] - cx) / fx, (px[1] - cy) / fy, 1.f);
      //Eigen::Vector3f pt3D = ray * depth;
      precomputed_rays[e->x + e->y * width] = Eigen::Vector3f((px[0] - cx) / fx, (px[1] - cy) / fy, 1.0f);
      Eigen::Vector3f pt3D = precomputed_rays[e->x + e->y * width] * depth;
      // Motion compensation
      Eigen::Vector3f pt_warped = pt3D;


      if (do_motion_correction && (!imu_sat || !gyro_sat))
      {
          //pt_warped = (1.f - dt) * pt3D + dt * (R * pt3D + t);
          //last_R = R;
          //last_t = t;

    Sophus::SE3 T_interp = Sophus::SE3::exp(dt * log_T);
    pt_warped = (T_interp * pt3D.template cast<double>()).template cast<float>();
  
          last_R = (T_interp.rotation_matrix()).template cast<float>(); //T_interp.so3().matrix();
          last_t = (T_interp.translation()).template cast<float>(); //T_interp.translation();
      }else{
          //pt_warped = (1.f - dt) * pt3D + dt * (last_R * pt3D + last_t);
          //pt_warped = T_interp * pt3D;
          pt_warped =  pt3D - last_t;
      }
      // Back-project to image plane
      float u = fx * pt_warped[0] / pt_warped[2] + cx;
      float v = fy * pt_warped[1] / pt_warped[2] + cy;

      // Save the event if valid
      if (std::isfinite(u) && std::isfinite(v)) {
          events.col(n_events++) << u, v;
      }
  }

  // Event frame’e event’leri ekle
  #pragma omp parallel for
  for (size_t i=0; i != n_events; ++i)
  {
    const Eigen::Vector2f& f = events.col(i);

    int x0 = std::floor(f[0]);
    int y0 = std::floor(f[1]);


    if(x0 >= 0 && x0 < width-1 && y0 >= 0 && y0 < height-1)
    {
      const float fx = f[0] - x0,
                  fy = f[1] - y0;
      Eigen::Vector4f w((1.f-fx)*(1.f-fy),
                        (fx)*(1.f-fy),
                        (1.f-fx)*(fy),
                        (fx)*(fy));

      out.at<float>(y0,   x0)     +=   w[0];
      out.at<float>(y0,   x0 + 1) +=   w[1];
      out.at<float>(y0 + 1, x0)   +=   w[2];
      out.at<float>(y0 + 1, x0 + 1) +=   w[3];
    }
  }

  //auto start_time = std::chrono::high_resolution_clock::now();

  // THIS PART IS REALLY IMPORTANT !!!!!! 
  //cv::threshold(out, out, 0.1, 255, cv::THRESH_TOZERO);
  //cv::normalize(out, out, 0, 255, cv::NORM_MINMAX);

  // ------------------PLOTTING PART -------------
  //cv::Mat visualized;
  //out.convertTo(visualized, CV_8U);  // float → 8-bit
  //cv::normalize(visualized, visualized, 0, 255, cv::NORM_MINMAX);
  //cv::imshow("Event Image Unprocessed", visualized);
  //cv::waitKey(10);
  //
  //static int frame_id = 0;
  //if(frame_id % 10 ==0)
  //{
  //  std::string filename = "/home/sebnem/event_frames/Unprocessed/event_" + std::to_string(frame_id) + ".png";
  //  cv::imwrite(filename, visualized);
  //} 
  //frame_id ++; 
  // ------------------------- Plotting Part end ---------------------------

  //

  // Normalize
  //cv::normalize(out, out, 0, 255, cv::NORM_MINMAX);
  cv::setUseOptimized(true);
  cv::setNumThreads(cv::getNumberOfCPUs());  // Tüm CPU çekirdeklerini kullan
  
   // Noise reduction
   cv::Mat blur_input,output_img;
   out.convertTo(blur_input, CV_8U);  // float → 8-bit
   cv::medianBlur(blur_input, blur_input, 3);  // artık çalışır
   //cv::GaussianBlur(blur_input, blur_input, cv::Size(3, 3), 0.8);

   // If you want to apply CLAHE you can open, but not recommended in real-time
   //cv::blur(blur_input, blur_input, cv::Size(3, 3));  // Ultra hızlı, çok hafif yumuşatma
   //cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.0, cv::Size(4, 4));
   //clahe->apply(blur_input, blur_input);


   // -------------- Plot and sace the frames -----------------------
   //cv::Mat visualized2;
   //blur_input.copyTo(visualized2);  // float → 8-bit
   //cv::normalize(visualized2, visualized2, 0, 255, cv::NORM_MINMAX);
   //cv::imshow("Event Image CLAHE", visualized2);
   //cv::waitKey(10);
//
   //if(frame_id % 10 ==0)
   //{
   //  std::string filename = "/home/sebnem/event_frames/CLAHE/event_" + std::to_string(frame_id) + ".png";
   //  cv::imwrite(filename, visualized2);
   //}
   // ---------------------- Plot and save the frames ------------------------

   
   

   //output_img.convertTo(out, CV_32F);  // geri float'a çevir istersen
  blur_input.convertTo(out, CV_32F);  // geri float'a çevir istersen
   
   //cv::normalize(out, out, 0, 255, cv::NORM_MINMAX);

  std::string filter_method = "Gaussian";

  cv::Mat processedInput;
  // cv::medianBlur(out, processedInput, 3);
  //cv::GaussianBlur(out, processedInput, cv::Size(3, 3), 0.8, 0.8);
  // Step 2: Gamma Correction
  if (filter_method =="Gaussian")
   {
    cv::Mat f_input,blur,input;
    out.convertTo(f_input, CV_32F, 1.0 / 255.0);
    cv::pow(f_input, 0.96, f_input);  // ayarlanabilir
    f_input.convertTo(input, CV_8U, 255.0);

    // Step 3: Unsharp Mask (yumuşak detay boost)
    cv::GaussianBlur(input, blur, cv::Size(3, 3), 1.0);
    cv::addWeighted(input, 0.9, blur, 0.1, 0, processedInput);
   }
   else{
      out.convertTo(processedInput,CV_8U);
   }
  //

  

  std::string method = "Canny";
    cv::Mat  edges, scharr_x, scharr_y, scharr_edges, thinned_edges, realFrame;
    //cv::medianBlur(input, processedInput, 3); // instead of gaussian, median blur is a more relavant solution for image noises

    if (method == "Canny") {
        cv::Mat blurred;

        // SCHAR based Canny  EDGE DEDECTOR is best for real time applications (too fast)
        cv::Mat grad_x, grad_y;
        cv::Scharr(processedInput, grad_x, CV_16S, 1, 0);
        cv::Scharr(processedInput, grad_y, CV_16S, 0, 1);
        cv::convertScaleAbs(grad_x, grad_x);
        cv::convertScaleAbs(grad_y, grad_y);
        cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, edges);
        //cv::Mat abs_grad_x, abs_grad_y;
        //cv::absdiff(grad_x, cv::Scalar::all(0), abs_grad_x);
        //cv::absdiff(grad_y, cv::Scalar::all(0), abs_grad_y);
        //cv::add(abs_grad_x, abs_grad_y, edges, cv::noArray(), CV_8U);

      }
    else if (method == "Laplacian") {
        cv::Mat input_32F;
        processedInput.convertTo(input_32F, CV_32F, 1.0 / 255.0);  // [0,1] aralığı
        cv::Laplacian(input_32F, edges, CV_32F, 3);
        edges.convertTo(edges, CV_8U);
    }
    else if (method == "Sobel") {
        cv::Mat sobel_x, sobel_y;
        cv::Mat input_32F;
        processedInput.convertTo(input_32F, CV_32F);
        cv::Sobel(input_32F, sobel_x, CV_32F, 1, 0, 3);
        cv::Sobel(input_32F, sobel_y, CV_32F, 0, 1, 3);
        cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, edges);
        // Normalize before converting to CV_8U (Prevents data loss)
        //cv::normalize(edges, edges, 0, 255, cv::NORM_MINMAX);
        edges.convertTo(edges, CV_8U);
    }
    else {
        std::cerr << "ERROR: Invalid method selected! Defaulting to Canny.\n";
        out.convertTo(out, CV_8U);
    }
  
    // ✅ Apply Scharr filter
    if (filter_method =="Scharr")
    {
    cv::Scharr(edges, scharr_x, CV_8U, 1, 0);
    cv::Scharr(edges, scharr_y, CV_8U, 0, 1);
    //cv::bitwise_or(scharr_x, scharr_y, scharr_edges); // Faster than addWeighted
    cv::addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0, thinned_edges);
    if (thinned_edges.empty()) {
        std::cerr << "WARNING: Thinned edges output is empty! Using edges instead." << std::endl;
        thinned_edges = edges.clone();
    }
     //✅ Normalize for better contrast
    cv::normalize(thinned_edges, thinned_edges, 0, 255, cv::NORM_MINMAX);
    } // Apply thinning using erosion

    if(filter_method =="Erosion")
    {
      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
      cv::erode(edges, thinned_edges, kernel, cv::Point(1, 1), 1);
    }
    else{
      edges.copyTo(thinned_edges);
      //cv::addWeighted(processedInput, 0.3, thinned_edges, 0.7, 0, out);
    }

  
    out.convertTo(out,CV_8U);
    cv::addWeighted(out, 0.7, thinned_edges, 0.3, 0, out);


    
   
}



/*
// --------------------------------------------- NEW CODE START - ONLY OPTICAL FLOW------------------------------------------------------------
void FrontendBase::drawEvents(
  const EventArray::iterator& first,
  const EventArray::iterator& last,
  const int64_t& t0,
  const int64_t& t1,
  const Transformation& T_1_0,
  cv::Mat &out)
{

  size_t n_events = 0;
  Eigen::Matrix<float, 2, Eigen::Dynamic> events;

  // Persistent previous and current event frames
  cv::Mat& prev_event_im = getPrevFrame();  // Get the persistent frame
  cv::Mat& event_img_ = getCurrentFrame();  // Get the persistent frame

  event_img_.convertTo(event_img_,CV_32F);

  //cv::Mat event_img_;  // Get the persistent frame
  events.resize(2, last - first);
  // DVS kamerayı alın
  Camera::ConstPtr dvs_cam;
  if (!rig_->getDvsCamera(&dvs_cam)) {
      LOG(WARNING) << "Draw events is called but there is no DVS camera in rig";
      return;
  }

  // Kamera boyutlarını alın
  const int height = dvs_cam->height();
  const int width = dvs_cam->width();

  CHECK_EQ(out.rows, height);
  CHECK_EQ(out.cols, width);

  const VectorX& params = dvs_cam->projectionParameters();
  const float fx = params[0];
  const float fy = params[1];
  const float cx = params[2];
  const float cy = params[3];
  Eigen::Matrix4f K;
  K << fx, 0., cx, 0.,
       0., fy, cy, 0.,
       0., 0., 1., 0.,
       0., 0., 0., 1.;


  // Çıkış matrisinin boyutlarını kontrol edin ve gerektiğinde yeniden boyutlandırın
  if (out.empty() || out.rows != height || out.cols != width) {
     out = cv::Mat::zeros(height, width, CV_32F);
  }


  if(event_img_.empty()|| event_img_.rows != height || event_img_.cols != width)
  { 
    event_img_ = cv::Mat::zeros(height, width, CV_32F);
    std::cerr << "[WARNING] event_img_ sıfırlandı." << std::endl;
  }


  //Eigen::Matrix4f T = K * T_1_0.getTransformationMatrix().cast<float>() * K.inverse();

  float depth = scene_depth_;

  bool do_motion_correction = FLAGS_vio_do_motion_correction;
  float dt = 0;

  
  //float dt = 0;
  Eigen::Matrix3f R = T_1_0.getRotation().cast<float>().toImplementation().toRotationMatrix();
  Eigen::Vector3f t = T_1_0.getPosition().cast<float>();


  static std::atomic<bool> imu_sat; 
  static std::atomic<bool> gyro_sat; 
  imu_sat = ze::nlls::ImuError::imu_saturated.load();
  gyro_sat = ze::nlls::ImuError::gyro_saturated.load();

  static Eigen::Matrix3f last_R = Eigen::Matrix3f::Identity();
  static Eigen::Vector3f last_t = Eigen::Vector3f::Zero();


  //instead of using linear interpolation use SE(3) monifold (realistic camera motion)
  Sophus::SE3 T_rel(R, t);                 // T_1_0 = T1 * T0⁻¹
  Sophus::SE3 T0 = Sophus::SE3();         // Identity
  Sophus::Tangent log_T = T_rel.log();  // sadece 1 kez

  for (auto e = first; e != last; ++e)
  {  // 1. kindr'dan rotation ve translation al (double)
      {
          dt = static_cast<float>(t1 - e->ts.toNSec()) / (t1 - t0);
          // add clamp 
      }

      // Original pixel coordinates
      Eigen::Vector2f px = rig_->dvs_keypoint_lut_.col(e->x + e->y * width);

      // Reproject into normalized camera coordinates
      Eigen::Vector3f ray((px[0] - cx) / fx, (px[1] - cy) / fy, 1.f);
      Eigen::Vector3f pt3D = ray * depth;
      // Motion compensation
      Eigen::Vector3f pt_warped = pt3D;
      imu_sat = ze::nlls::ImuError::imu_saturated.load();
      gyro_sat = ze::nlls::ImuError::gyro_saturated.load();


      if (do_motion_correction && (!imu_sat || !gyro_sat))
      {
          //pt_warped = (1.f - dt) * pt3D + dt * (R * pt3D + t);
          //last_R = R;
          //last_t = t;

          Sophus::SE3 T_interp = Sophus::SE3::exp(dt * log_T);
          pt_warped = T_interp * pt3D;
  
          last_R = T_interp.so3().matrix();
          last_t = T_interp.translation();
      }else{
          //pt_warped = (1.f - dt) * pt3D + dt * (last_R * pt3D + last_t);
          //Sophus::SE3 T_interp = Sophus::SE3::exp(dt * log_T);
          //pt_warped = T_interp * pt3D;
          pt_warped =  pt3D - last_t;
      }
      // Back-project to image plane
      float u = fx * pt_warped[0] / pt_warped[2] + cx;
      float v = fy * pt_warped[1] / pt_warped[2] + cy;

      // Save the event if valid
      if (std::isfinite(u) && std::isfinite(v)) {
          events.col(n_events++) << u, v;
      }
  }

  // Event frame’e event’leri ekle
  for (size_t i=0; i != n_events; ++i)
  {
    const Eigen::Vector2f& f = events.col(i);

    int x0 = std::floor(f[0]);
    int y0 = std::floor(f[1]);

    // ⚠️ Yeni: Polariteye göre ağırlık belirle
    float polarity_weight = (first + i)->polarity ? +1.0f : -1.0f;

    if(x0 >= 0 && x0 < width-1 && y0 >= 0 && y0 < height-1)
    {
      const float fx = f[0] - x0,
                  fy = f[1] - y0;
      Eigen::Vector4f w((1.f-fx)*(1.f-fy),
                        (fx)*(1.f-fy),
                        (1.f-fx)*(fy),
                        (fx)*(fy));

      event_img_.at<float>(y0,   x0)     += polarity_weight * w[0];
      event_img_.at<float>(y0,   x0 + 1) += polarity_weight * w[1];
      event_img_.at<float>(y0 + 1, x0)   += polarity_weight * w[2];
      event_img_.at<float>(y0 + 1, x0 + 1) += polarity_weight * w[3];
    }
  }

   // Event frame'ini normalize et ve 8-bit görüntüye dönüştür

  //event_img_.convertTo(event_img_,CV_8U);
   // Convert to 8-bit image for visualization

   cv::normalize(event_img_, event_img_, 0, 255, cv::NORM_MINMAX);
   //event_img_.convertTo(event_img_, CV_8U);


   // Ensure prev_event_im is properly initialized
   if (prev_event_im.empty() ) { 
     event_img_.copyTo(prev_event_im);
   }
   //prev_event_im.convertTo(prev_event_im,CV_8U);

   if (prev_event_im.size() != event_img_.size()) {
    std::cerr << "[ERROR] prev_event_im and event_img_ size mismatch: "
              << prev_event_im.size() << " vs. " << event_img_.size() << std::endl;
    prev_event_im = cv::Mat::zeros(event_img_.size(), CV_8U);
    }


  // Optical flow için noktaları hazırla
  std::vector<cv::Point2f> prev_points, curr_points;
  std::vector<uchar> status;
  std::vector<float> err;

  #pragma omp parallel for
  //for (auto e = first; e != last; ++e) {
  //    cv::Point2f pt(e->x, e->y);
  //    prev_points.push_back(pt);
  //}

  int step = 1;
  int i = 0;
  for (auto e = first; e != last; ++e, ++i) {
      if (i % step == 0) {
          prev_points.emplace_back(e->x, e->y);
      }
  }

    // ıf you want u can use gaussian filter
    std::string filter_method = "";

    // cv::medianBlur(out, processedInput, 3);
    // Step 2: Gamma Correction
    if (filter_method =="Gaussian")
     {
      cv::Mat f_input,blur,input,sharp_result;
      cv::Mat blurred, sharp;
      event_img_.convertTo(event_img_,CV_8U);
      cv::GaussianBlur(event_img_, blurred, cv::Size(0, 0), 1.0);
      cv::addWeighted(event_img_, 1.5, blurred, -0.5, 0, event_img_);

  
      // Step 3: Unsharp Mask (yumuşak detay boost)
      //input.convertTo(event_img_,CV_8U);
     }
     else{
      event_img_.convertTo(event_img_,CV_8U);
     }


     prev_event_im.convertTo(prev_event_im, CV_8U);


  // Optical flow hesapla
  if (!prev_points.empty() && !event_img_.empty() && !prev_points.empty()) {
      
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.1);
    cv::Size winSize(7, 7);
    int maxLevel = 0;

  
    try {
      cv::calcOpticalFlowPyrLK(
          prev_event_im, event_img_, prev_points, curr_points,
          status, err, winSize, maxLevel, termcrit, 0, 0.001);
    } catch (const cv::Exception& e) {
        std::cerr << "Optical flow hatası: " << e.what() << std::endl;
        return;
    }

    std::vector<cv::Mat> thread_out(omp_get_max_threads(), cv::Mat::zeros(out.size(), CV_32F));

    
      // Optical flow ile taşınan event'leri işle
      #pragma omp parallel for
      for (size_t i = 0; i < prev_points.size(); i=i+1) {

        //int thread_id = omp_get_thread_num(); // Get thread ID
        //cv::Mat &local_out = thread_out[thread_id]; // Each thread gets its own matrix
        // ⚠️ Yeni: Polariteye göre ağırlık belirle
        float polarity_weight = (first + i)->polarity ? +1.0f : -1.0f;

        if (std::isnan(curr_points[i].x) || std::isnan(curr_points[i].y) || std::isinf(curr_points[i].x) || std::isinf(curr_points[i].y)) 
        {
              std::cerr << "[ERROR] Optical flow hatalı nokta: (" 
              << curr_points[i].x << ", " << curr_points[i].y << ")" << std::endl;
              curr_points[i] = prev_points[i];  // Eski noktayı kullanarak düzelt
         }

         float flow_len = cv::norm(prev_points[i] - curr_points[i]);
          if (status[i] && err[i] < 15.0f)
           {
              cv::Point2f new_pos = curr_points[i];

              Eigen::Vector4f f;
              f.head<2>() << new_pos.x, new_pos.y;
              f[2] = 1.;
              f[3] = 1. / scene_depth_;

              // Event'leri frame'e çiz
              int x0 = std::round(f[0]);
              int y0 = std::round(f[1]);

              if (x0 >= 0 && x0 < width - 1 && y0 >= 0 && y0 < height - 1) {
                  
                  const float fx = f[0] - x0;
                  const float fy = f[1] - y0;
                  Eigen::Vector4f w((1.f - fx) * (1.f - fy),
                                    (fx) * (1.f - fy),
                                    (1.f - fx) * (fy),
                                    (fx) * (fy));

                  // Event yoğunluğunu çiz
                  out.at<float>(y0, x0)     +=  w[0];
                  out.at<float>(y0, x0 + 1) +=  w[1];
                  out.at<float>(y0 + 1, x0) +=  w[2];
                  out.at<float>(y0 + 1, x0 + 1) += w[3];
                  
                  //out.at<float>(y0, x0) += 1.0f;
              }
          }
      }
  }
  else{
    event_img_.copyTo(out);
  }

  cv::Mat f_input,blur,input;
  out.convertTo(f_input, CV_32F, 1.0 / 255.0);
  cv::pow(f_input, 0.96, f_input);  // ayarlanabilir
  f_input.convertTo(input, CV_8U, 255.0);

  // Step 3: Unsharp Mask (yumuşak detay boost)
  cv::GaussianBlur(input, blur, cv::Size(0, 0), 1.0);
  cv::addWeighted(input, 1.3, blur, -0.3, 0, out);


  // Assign the old frame as the prev event image
  if (prev_event_im.empty() || cv::countNonZero(prev_event_im) == 0 || event_img_.size() != prev_event_im.size()) {
    prev_event_im = cv::Mat::zeros(event_img_.size(), CV_8U); // Siyah bir frame ile başlat
  }
  else {
    //prev_event_im = event_img_;  // Optical flow'da eski frame'in gerçekten değiştiğinden emin ol
    //event_img_.copyTo(prev_event_im);
    event_img_.copyTo(prev_event_im);
    
  }
  
   
}*/

// FUNCTION for dataset performance analysis
// Motion Compensation with Optical Flow (Lucas-Kanade)
/**/
/*
// --------------------------------------------- NEW CODE START ------------------------------------------------------------
void FrontendBase::drawEvents(
  const EventArray::iterator& first,
  const EventArray::iterator& last,
  const int64_t& t0,
  const int64_t& t1,
  const Transformation& T_1_0,
  cv::Mat &out)
{

  size_t n_events = 0;
  Eigen::Matrix<float, 2, Eigen::Dynamic> events;

  // Persistent previous and current event frames
  cv::Mat& prev_event_im = getPrevFrame();  // Get the persistent frame
  cv::Mat& event_img_ = getCurrentFrame();  // Get the persistent frame

  event_img_.convertTo(event_img_,CV_32F);

  //cv::Mat event_img_;  // Get the persistent frame
  events.resize(2, last - first);
  // DVS kamerayı alın
  Camera::ConstPtr dvs_cam;
  if (!rig_->getDvsCamera(&dvs_cam)) {
      LOG(WARNING) << "Draw events is called but there is no DVS camera in rig";
      return;
  }

  // Kamera boyutlarını alın
  const int height = dvs_cam->height();
  const int width = dvs_cam->width();

  CHECK_EQ(out.rows, height);
  CHECK_EQ(out.cols, width);

  const VectorX& params = dvs_cam->projectionParameters();
  const float fx = params[0];
  const float fy = params[1];
  const float cx = params[2];
  const float cy = params[3];
  Eigen::Matrix4f K;
  K << fx, 0., cx, 0.,
       0., fy, cy, 0.,
       0., 0., 1., 0.,
       0., 0., 0., 1.;


  // Çıkış matrisinin boyutlarını kontrol edin ve gerektiğinde yeniden boyutlandırın
  if (out.empty() || out.rows != height || out.cols != width) {
     out = cv::Mat::zeros(height, width, CV_32F);
  }


  if(event_img_.empty()|| event_img_.rows != height || event_img_.cols != width)
  { 
    event_img_ = cv::Mat::zeros(height, width, CV_32F);
    std::cerr << "[WARNING] event_img_ sıfırlandı." << std::endl;
  }


  //Eigen::Matrix4f T = K * T_1_0.getTransformationMatrix().cast<float>() * K.inverse();

  float depth = scene_depth_;

  bool do_motion_correction = FLAGS_vio_do_motion_correction;
  float dt = 0;

  
  Eigen::Matrix3f R = T_1_0.getRotation().cast<float>().toImplementation().toRotationMatrix();
  Eigen::Vector3f t = T_1_0.getPosition().cast<float>();


  static std::atomic<bool> imu_sat; 
  static std::atomic<bool> gyro_sat; 
  imu_sat = ze::nlls::ImuError::imu_saturated.load();
  gyro_sat = ze::nlls::ImuError::gyro_saturated.load();

  static Eigen::Matrix3f last_R = Eigen::Matrix3f::Identity();
  static Eigen::Vector3f last_t = Eigen::Vector3f::Zero();


  //instead of using linear interpolation use SE(3) monifold (realistic camera motion)
  Sophus::SE3 T_rel(R, t);                 // T_1_0 = T1 * T0⁻¹
  Sophus::SE3 T0 = Sophus::SE3();         // Identity
  Sophus::Tangent log_T = T_rel.log();  // sadece 1 kez

  for (auto e = first; e != last; ++e)
  {  // 1. kindr'dan rotation ve translation al (double)
      {
          dt = static_cast<float>(t1 - e->ts.toNSec()) / (t1 - t0);
          // add clamp 
      }

      // Original pixel coordinates
      Eigen::Vector2f px = rig_->dvs_keypoint_lut_.col(e->x + e->y * width);

      // Reproject into normalized camera coordinates
      Eigen::Vector3f ray((px[0] - cx) / fx, (px[1] - cy) / fy, 1.f);
      Eigen::Vector3f pt3D = ray * depth;
      // Motion compensation
      Eigen::Vector3f pt_warped = pt3D;
      imu_sat = ze::nlls::ImuError::imu_saturated.load();
      gyro_sat = ze::nlls::ImuError::gyro_saturated.load();


      if (do_motion_correction && (!imu_sat || !gyro_sat))
      {
          //pt_warped = (1.f - dt) * pt3D + dt * (R * pt3D + t);
          //last_R = R;
          //last_t = t;

          Sophus::SE3 T_interp = Sophus::SE3::exp(dt * log_T);
          pt_warped = T_interp * pt3D;
  
          last_R = T_interp.so3().matrix();
          last_t = T_interp.translation();
      }else{
          //pt_warped = (1.f - dt) * pt3D + dt * (last_R * pt3D + last_t);
          Sophus::SE3 T_interp = Sophus::SE3::exp(dt * log_T);
          pt_warped = T_interp * pt3D;
      }
      // Back-project to image plane
      float u = fx * pt_warped[0] / pt_warped[2] + cx;
      float v = fy * pt_warped[1] / pt_warped[2] + cy;

      // Save the event if valid
      if (std::isfinite(u) && std::isfinite(v)) {
          events.col(n_events++) << u, v;
      }
  }


  // Event frame’e event’leri ekle
  for (size_t i=0; i != n_events; ++i)
  {
    const Eigen::Vector2f& f = events.col(i);

    int x0 = std::floor(f[0]);
    int y0 = std::floor(f[1]);

    // ⚠️ Yeni: Polariteye göre ağırlık belirle
    float polarity_weight = (first + i)->polarity ? +1.0f : -1.0f;

    if(x0 >= 0 && x0 < width-1 && y0 >= 0 && y0 < height-1)
    {
      const float fx = f[0] - x0,
                  fy = f[1] - y0;
      Eigen::Vector4f w((1.f-fx)*(1.f-fy),
                        (fx)*(1.f-fy),
                        (1.f-fx)*(fy),
                        (fx)*(fy));

      event_img_.at<float>(y0,   x0)     += polarity_weight * w[0];
      event_img_.at<float>(y0,   x0 + 1) += polarity_weight * w[1];
      event_img_.at<float>(y0 + 1, x0)   += polarity_weight * w[2];
      event_img_.at<float>(y0 + 1, x0 + 1) += polarity_weight * w[3];
    }
  }
  

  // Event frame'ini normalize et ve 8-bit görüntüye dönüştür

  //event_img_.convertTo(event_img_,CV_8U);
   // Convert to 8-bit image for visualization

  cv::normalize(event_img_, event_img_, 0, 255, cv::NORM_MINMAX);
   event_img_.convertTo(event_img_, CV_8U);


   // Ensure prev_event_im is properly initialized
   if (prev_event_im.empty() ) { 
     event_img_.copyTo(prev_event_im);
   }
   //prev_event_im.convertTo(prev_event_im,CV_8U);

   if (prev_event_im.size() != event_img_.size()) {
    std::cerr << "[ERROR] prev_event_im and event_img_ size mismatch: "
              << prev_event_im.size() << " vs. " << event_img_.size() << std::endl;
    prev_event_im = cv::Mat::zeros(event_img_.size(), CV_8U);
    }


  // Optical flow için noktaları hazırla
  std::vector<cv::Point2f> prev_points, curr_points;
  std::vector<uchar> status;
  std::vector<float> err;

  #pragma omp parallel for
  //for (auto e = first; e != last; ++e) {
  //    cv::Point2f pt(e->x, e->y);
  //    prev_points.push_back(pt);
  //}

  int step = 1;
  int i = 0;
  for (auto e = first; e != last; ++e, ++i) {
      if (i % step == 0) {
          prev_points.emplace_back(e->x, e->y);
      }
  }


  // Optical flow hesapla
  if (!prev_points.empty() && !event_img_.empty() && !prev_points.empty() && (!imu_sat || !gyro_sat )) {
      
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 5, 0.1);
    cv::Size winSize(7, 7);
    int maxLevel = 0;

  
    try {
      cv::calcOpticalFlowPyrLK(
          prev_event_im, event_img_, prev_points, curr_points,
          status, err, winSize, maxLevel, termcrit, 0, 0.001);
    } catch (const cv::Exception& e) {
        std::cerr << "Optical flow hatası: " << e.what() << std::endl;
        return;
    }

    std::vector<cv::Mat> thread_out(omp_get_max_threads(), cv::Mat::zeros(out.size(), CV_32F));

    
      // Optical flow ile taşınan event'leri işle
      #pragma omp parallel for
      for (size_t i = 0; i < prev_points.size(); i=i+1) {

        int thread_id = omp_get_thread_num(); // Get thread ID
        cv::Mat &local_out = thread_out[thread_id]; // Each thread gets its own matrix
        // ⚠️ Yeni: Polariteye göre ağırlık belirle
        float polarity_weight = (first + i)->polarity ? +1.0f : -1.0f;

        if (std::isnan(curr_points[i].x) || std::isnan(curr_points[i].y) || std::isinf(curr_points[i].x) || std::isinf(curr_points[i].y)) 
        {
              std::cerr << "[ERROR] Optical flow hatalı nokta: (" 
              << curr_points[i].x << ", " << curr_points[i].y << ")" << std::endl;
              curr_points[i] = prev_points[i];  // Eski noktayı kullanarak düzelt
         }

         float flow_len = cv::norm(prev_points[i] - curr_points[i]);
          if (status[i] && err[i] < 15.0f)
           {
              cv::Point2f new_pos = curr_points[i];

              Eigen::Vector4f f;
              f.head<2>() << new_pos.x, new_pos.y;
              f[2] = 1.;
              f[3] = 1. / scene_depth_;

              // Event'leri frame'e çiz
              int x0 = std::round(f[0]);
              int y0 = std::round(f[1]);

              if (x0 >= 0 && x0 < width - 1 && y0 >= 0 && y0 < height - 1) {
                  
                  const float fx = f[0] - x0;
                  const float fy = f[1] - y0;
                  Eigen::Vector4f w((1.f - fx) * (1.f - fy),
                                    (fx) * (1.f - fy),
                                    (1.f - fx) * (fy),
                                    (fx) * (fy));

                  // Event yoğunluğunu çiz
                  local_out.at<float>(y0, x0)     +=  w[0];
                  local_out.at<float>(y0, x0 + 1) +=  w[1];
                  local_out.at<float>(y0 + 1, x0) +=  w[2];
                  local_out.at<float>(y0 + 1, x0 + 1) += w[3];
                  
                  //out.at<float>(y0, x0) += 1.0f;
              }
          }
      }
      for (const auto& mat : thread_out) {
        out += mat;
      }
    
  }
  else{
    event_img_.copyTo(out);
  }


  //out.convertTo(out, CV_8U, 255.0);
  /// Texture Enchacement 
  out.convertTo(out, CV_8U, 255.0);
 //// 1. Gabor filtre ile doku belirginleştirme
 //cv::Mat gabor_output;
 //double sigma = 1.2;
 //double theta = CV_PI / 4;
 //double lambd = 100.0;
 //double gamma = 0.5;
 //double psi = 0;
 //int kernel_size = 3;
 //
 //cv::Mat gabor_kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, CV_32F);
 //
 //// Gabor'u processedInput üzerine uygula (medianBlur sonrası frame)
 //cv::filter2D(out, gabor_output, CV_32F, gabor_kernel);
 //cv::convertScaleAbs(gabor_output, gabor_output);
 //
 //// Blend: Orijinal + Gabor ile dokular zenginleştirilsin
 //cv::Mat enhanced_input;
 //cv::addWeighted(out, 0.7, gabor_output, 0.3, 0, enhanced_input);
// Step 1: Median Blur

std::string filter_method = "Erosion";

  cv::Mat processedInput;
  // cv::medianBlur(out, processedInput, 3);
  cv::GaussianBlur(out, processedInput, cv::Size(3, 3), 0.8, 0.8);
  // Step 2: Gamma Correction
  if (filter_method =="Gaussian")
   {
    cv::Mat f_input,blur,input;
    out.convertTo(f_input, CV_32F, 1.0 / 255.0);
    cv::pow(f_input, 0.95, f_input);  // ayarlanabilir
    f_input.convertTo(input, CV_8U, 255.0);

    // Step 3: Unsharp Mask (yumuşak detay boost)
    cv::GaussianBlur(input, blur, cv::Size(0, 0), 1.0);
    cv::addWeighted(input, 0.9, blur, 0.1, 0, processedInput);
   }
   else{
      out.copyTo(processedInput);
   }
  //

  
  std::string method = "Laplacian";
  //applyEdgeDetection(out, method); 
  //output.copyTo(out);
    // ✅ Make sure input is not modified in-place
    //cv::Mat img = out.clone(); //= input;
    cv::Mat  edges, scharr_x, scharr_y, scharr_edges, thinned_edges, realFrame;
    //cv::medianBlur(input, processedInput, 3); // instead of gaussian, median blur is a more relavant solution for image noises

    if (method == "Canny") {
        cv::Mat blurred;
        double sigma = 0.33;
        double median_val = cv::mean(processedInput)[0];
        double low_thresh = std::max(50.0, (1.0 - sigma) * median_val);
        double high_thresh = std::max(150.0, (1.0 + sigma) * median_val);
  
        cv::Canny(processedInput, edges, low_thresh, high_thresh);
    }
    else if (method == "Laplacian") {
        cv::Mat input_32F;
        processedInput.convertTo(input_32F, CV_32F, 1.0 / 255.0);  // [0,1] aralığı
        cv::Laplacian(input_32F, edges, CV_32F, 3);
        edges.convertTo(edges, CV_8U);
    }
    else if (method == "Sobel") {
        cv::Mat sobel_x, sobel_y;
        processedInput.convertTo(processedInput, CV_32F);
        cv::Sobel(processedInput, sobel_x, CV_32F, 1, 0, 3);
        cv::Sobel(processedInput, sobel_y, CV_32F, 0, 1, 3);
        cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, edges);
        // ✅ Normalize before converting to CV_8U (Prevents data loss)
        cv::normalize(edges, edges, 0, 255, cv::NORM_MINMAX);
        edges.convertTo(edges, CV_8U);
    }
    else {
        std::cerr << "ERROR: Invalid method selected! Defaulting to Canny.\n";
        out.convertTo(out, CV_8U);
        cv::Canny(out, edges, 50, 150);
    }
  
    // ✅ Ensure edges exist before further processing
    if (edges.empty()) {
        std::cerr << "WARNING: Edge detection output is empty!" << std::endl;
        out = cv::Mat::zeros(out.size(), CV_8U);
        return;
    }
  
    // ✅ Apply Scharr filter
    if (filter_method =="Scharr")
    {
    cv::Scharr(edges, scharr_x, CV_8U, 1, 0);
    cv::Scharr(edges, scharr_y, CV_8U, 0, 1);
    //cv::bitwise_or(scharr_x, scharr_y, scharr_edges); // Faster than addWeighted
    cv::addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0, thinned_edges);
    // ✅ Apply controlled thinning using erosion
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    //cv::erode(scharr_edges, thinned_edges, kernel, cv::Point(-1, -1), 1);
    //cv::dilate(edges, thinned_edges, kernel);  // Much faster than erosion
    //edges.convertTo(thinned_edges, CV_8U); // open if you are close the Scharr filter
     //✅ Ensure thinning result is valid
    if (thinned_edges.empty()) {
        std::cerr << "WARNING: Thinned edges output is empty! Using edges instead." << std::endl;
        thinned_edges = edges.clone();
    }
    thinned_edges.convertTo(thinned_edges, CV_8U);
     //✅ Normalize for better contrast
    cv::normalize(thinned_edges, out, 0, 255, cv::NORM_MINMAX);
    } // Apply thinning using erosion
    if(filter_method =="Erosion")
    {
      cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
      cv::erode(edges, thinned_edges, kernel, cv::Point(-1, -1), 1);
    }
    else{
      edges.copyTo(thinned_edges);
      cv::addWeighted(processedInput, 0.3, thinned_edges, 0.7, 0, out);
    }
  
    // ✅ Blend edges with the original image
    cv::addWeighted(processedInput, 0.3, thinned_edges, 0.7, 0, out);
    //cv::bitwise_or(out, thinned_edges, out);
   


  if (prev_event_im.empty() || cv::countNonZero(prev_event_im) == 0 || event_img_.size() != prev_event_im.size()) {
      prev_event_im = cv::Mat::zeros(event_img_.size(), CV_8U); // Siyah bir frame ile başlat
    }
    else {
      //prev_event_im = event_img_;  // Optical flow'da eski frame'in gerçekten değiştiğinden emin ol
      //event_img_.copyTo(prev_event_im);
      event_img_.copyTo(prev_event_im);
      
    }

}*/


// Motion Compensation with Optical Flow (Lucas-Kanade)
/**/
/*
// --------------------------------------------- NEW CODE START ------------------------------------------------------------
void FrontendBase::drawEvents(
  const EventArray::iterator& first,
  const EventArray::iterator& last,
  const int64_t& t0,
  const int64_t& t1,
  const Transformation& T_1_0,
  cv::Mat &out)
{

  size_t n_events = 0;
  Eigen::Matrix<float, 2, Eigen::Dynamic> events;

  // Persistent previous and current event frames
  cv::Mat& prev_event_im = getPrevFrame();  // Get the persistent frame
  cv::Mat& event_img_ = getCurrentFrame();  // Get the persistent frame

  event_img_.convertTo(event_img_,CV_32F);

  //cv::Mat event_img_;  // Get the persistent frame
  events.resize(2, last - first);
  // DVS kamerayı alın
  Camera::ConstPtr dvs_cam;
  if (!rig_->getDvsCamera(&dvs_cam)) {
      LOG(WARNING) << "Draw events is called but there is no DVS camera in rig";
      return;
  }

  // Kamera boyutlarını alın
  const int height = dvs_cam->height();
  const int width = dvs_cam->width();

  CHECK_EQ(out.rows, height);
  CHECK_EQ(out.cols, width);

  const VectorX& params = dvs_cam->projectionParameters();
  const float fx = params[0];
  const float fy = params[1];
  const float cx = params[2];
  const float cy = params[3];
  Eigen::Matrix4f K;
  K << fx, 0., cx, 0.,
       0., fy, cy, 0.,
       0., 0., 1., 0.,
       0., 0., 0., 1.;


  // Çıkış matrisinin boyutlarını kontrol edin ve gerektiğinde yeniden boyutlandırın
  if (out.empty() || out.rows != height || out.cols != width) {
     out = cv::Mat::zeros(height, width, CV_32F);
  }


  if(event_img_.empty()|| event_img_.rows != height || event_img_.cols != width)
  { 
    event_img_ = cv::Mat::zeros(height, width, CV_32F);
    std::cerr << "[WARNING] event_img_ sıfırlandı." << std::endl;
  }


  //Eigen::Matrix4f T = K * T_1_0.getTransformationMatrix().cast<float>() * K.inverse();

  float depth = scene_depth_;

  bool do_motion_correction = FLAGS_vio_do_motion_correction;
  float dt = 0;

  Eigen::Matrix3f R = T_1_0.getRotation().cast<float>().toImplementation().toRotationMatrix();
  Eigen::Vector3f t = T_1_0.getPosition().cast<float>();
  static std::atomic<bool> imu_sat; 
  static std::atomic<bool> gyro_sat; 
  imu_sat = ze::nlls::ImuError::imu_saturated.load();
  gyro_sat = ze::nlls::ImuError::gyro_saturated.load();
  

  for (auto e = first; e != last; ++e)
  {
      if (n_events % 10 == 0)
      {
          dt = static_cast<float>(t1 - e->ts.toNSec()) / (t1 - t0);
      }

      // Original pixel coordinates
      Eigen::Vector2f px = rig_->dvs_keypoint_lut_.col(e->x + e->y * width);

      // Reproject into normalized camera coordinates
      Eigen::Vector3f ray((px[0] - cx) / fx, (px[1] - cy) / fy, 1.f);
      Eigen::Vector3f pt3D = ray * depth;
      // Motion compensation
      Eigen::Vector3f pt_warped = pt3D;
      imu_sat = ze::nlls::ImuError::imu_saturated.load();
      gyro_sat = ze::nlls::ImuError::gyro_saturated.load();
      if (do_motion_correction && (!imu_sat || !gyro_sat))
      {
          pt_warped = (1.f - dt) * pt3D + dt * (R * pt3D + t);
      }else{
        pt_warped = pt3D;
      }
      // Back-project to image plane
      float u = fx * pt_warped[0] / pt_warped[2] + cx;
      float v = fy * pt_warped[1] / pt_warped[2] + cy;

      // Save the event if valid
      if (std::isfinite(u) && std::isfinite(v)) {
          events.col(n_events++) << u, v;
      }
  }


  // Event frame’e event’leri ekle
  for (size_t i=0; i != n_events; ++i)
  {
    const Eigen::Vector2f& f = events.col(i);

    int x0 = std::floor(f[0]);
    int y0 = std::floor(f[1]);

    // ⚠️ Yeni: Polariteye göre ağırlık belirle
    float polarity_weight = (first + i)->polarity ? +1.0f : -1.0f;

    if(x0 >= 0 && x0 < width-1 && y0 >= 0 && y0 < height-1)
    {
      const float fx = f[0] - x0,
                  fy = f[1] - y0;
      Eigen::Vector4f w((1.f-fx)*(1.f-fy),
                        (fx)*(1.f-fy),
                        (1.f-fx)*(fy),
                        (fx)*(fy));

      event_img_.at<float>(y0,   x0)     += polarity_weight * w[0];
      event_img_.at<float>(y0,   x0 + 1) += polarity_weight * w[1];
      event_img_.at<float>(y0 + 1, x0)   += polarity_weight * w[2];
      event_img_.at<float>(y0 + 1, x0 + 1) += polarity_weight * w[3];
    }
  }
  

  // Event frame'ini normalize et ve 8-bit görüntüye dönüştür

  //event_img_.convertTo(event_img_,CV_8U);
   // Convert to 8-bit image for visualization

  cv::normalize(event_img_, event_img_, 0, 255, cv::NORM_MINMAX);
   event_img_.convertTo(event_img_, CV_8U);


   // Ensure prev_event_im is properly initialized
   if (prev_event_im.empty() ) { 
     event_img_.copyTo(prev_event_im);
   }
   //prev_event_im.convertTo(prev_event_im,CV_8U);

   if (prev_event_im.size() != event_img_.size()) {
    std::cerr << "[ERROR] prev_event_im and event_img_ size mismatch: "
              << prev_event_im.size() << " vs. " << event_img_.size() << std::endl;
    prev_event_im = cv::Mat::zeros(event_img_.size(), CV_8U);
    }


  // Optical flow için noktaları hazırla
  std::vector<cv::Point2f> prev_points, curr_points;
  std::vector<uchar> status;
  std::vector<float> err;

  #pragma omp parallel for
  //for (auto e = first; e != last; ++e) {
  //    cv::Point2f pt(e->x, e->y);
  //    prev_points.push_back(pt);
  //}

  int step = 1;
  int i = 0;
  for (auto e = first; e != last; ++e, ++i) {
      if (i % step == 0) {
          prev_points.emplace_back(e->x, e->y);
      }
  }


  // Optical flow hesapla
  if (!prev_points.empty() && !event_img_.empty() && !prev_points.empty() && (!imu_sat || !gyro_sat )) {
      
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.01);
    cv::Size winSize(7, 7);
    int maxLevel = 0;

  
    try {
      cv::calcOpticalFlowPyrLK(
          prev_event_im, event_img_, prev_points, curr_points,
          status, err, winSize, maxLevel, termcrit, 0, 0.001);
    } catch (const cv::Exception& e) {
        std::cerr << "Optical flow hatası: " << e.what() << std::endl;
        return;
    }

    std::vector<cv::Mat> thread_out(omp_get_max_threads(), cv::Mat::zeros(out.size(), CV_32F));

    
      // Optical flow ile taşınan event'leri işle
      #pragma omp parallel for
      for (size_t i = 0; i < prev_points.size(); i=i+1) {

        int thread_id = omp_get_thread_num(); // Get thread ID
        cv::Mat &local_out = thread_out[thread_id]; // Each thread gets its own matrix


        if (std::isnan(curr_points[i].x) || std::isnan(curr_points[i].y) || std::isinf(curr_points[i].x) || std::isinf(curr_points[i].y)) 
        {
              std::cerr << "[ERROR] Optical flow hatalı nokta: (" 
              << curr_points[i].x << ", " << curr_points[i].y << ")" << std::endl;
              curr_points[i] = prev_points[i];  // Eski noktayı kullanarak düzelt
         }

         float flow_len = cv::norm(prev_points[i] - curr_points[i]);
          if (status[i] && err[i] < 15.0f)
           {
              cv::Point2f new_pos = curr_points[i];

              Eigen::Vector4f f;
              f.head<2>() << new_pos.x, new_pos.y;
              f[2] = 1.;
              f[3] = 1. / scene_depth_;

              // Event'leri frame'e çiz
              int x0 = std::round(f[0]);
              int y0 = std::round(f[1]);

              if (x0 >= 0 && x0 < width - 1 && y0 >= 0 && y0 < height - 1) {
                  
                  const float fx = f[0] - x0;
                  const float fy = f[1] - y0;
                  Eigen::Vector4f w((1.f - fx) * (1.f - fy),
                                    (fx) * (1.f - fy),
                                    (1.f - fx) * (fy),
                                    (fx) * (fy));

                  // Event yoğunluğunu çiz
                  local_out.at<float>(y0, x0)     += w[0];
                  local_out.at<float>(y0, x0 + 1) += w[1];
                  local_out.at<float>(y0 + 1, x0) += w[2];
                  local_out.at<float>(y0 + 1, x0 + 1) += w[3];
                  
                  //out.at<float>(y0, x0) += 1.0f;
              }
          }
      }
      for (const auto& mat : thread_out) {
        out += mat;
      }
    
  }
  else{
    event_img_.copyTo(out);
  }


  //out.convertTo(out, CV_8U, 255.0);
  /// Texture Enchacement 
  out.convertTo(out, CV_8U, 255.0);
 //// 1. Gabor filtre ile doku belirginleştirme
 //cv::Mat gabor_output;
 //double sigma = 1.2;
 //double theta = CV_PI / 4;
 //double lambd = 100.0;
 //double gamma = 0.5;
 //double psi = 0;
 //int kernel_size = 3;
 //
 //cv::Mat gabor_kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, CV_32F);
 //
 //// Gabor'u processedInput üzerine uygula (medianBlur sonrası frame)
 //cv::filter2D(out, gabor_output, CV_32F, gabor_kernel);
 //cv::convertScaleAbs(gabor_output, gabor_output);
 //
 //// Blend: Orijinal + Gabor ile dokular zenginleştirilsin
 //cv::Mat enhanced_input;
 //cv::addWeighted(out, 0.7, gabor_output, 0.3, 0, enhanced_input);
// Step 1: Median Blur

  cv::Mat processedInput;
  // cv::medianBlur(out, processedInput, 3);
  //
  // Step 2: Gamma Correction
  cv::Mat f_input,blur,input;
  out.convertTo(f_input, CV_32F, 1.0 / 255.0);
  cv::pow(f_input, 0.95, f_input);  // ayarlanabilir
  f_input.convertTo(input, CV_8U, 255.0);
  
  // Step 3: Unsharp Mask (yumuşak detay boost)
  cv::GaussianBlur(input, blur, cv::Size(0, 0), 1.0);
  cv::addWeighted(input, 0.9, blur, 0.1, 0, processedInput);
  //

  
  std::string method = "Canny";
  //applyEdgeDetection(out, method); 
  //output.copyTo(out);
    // ✅ Make sure input is not modified in-place
    //cv::Mat img = out.clone(); //= input;
    cv::Mat  edges, scharr_x, scharr_y, scharr_edges, thinned_edges, realFrame;
    //cv::medianBlur(input, processedInput, 3); // instead of gaussian, median blur is a more relavant solution for image noises

    if (method == "Canny") {
        cv::Mat blurred;
        double sigma = 0.33;
        double median_val = cv::mean(processedInput)[0];
        double low_thresh = std::max(50.0, (1.0 - sigma) * median_val);
        double high_thresh = std::max(150.0, (1.0 + sigma) * median_val);
  
        cv::Canny(processedInput, edges, low_thresh, high_thresh);
    }
    else if (method == "Laplacian") {
        processedInput.convertTo(processedInput, CV_32F);
        cv::Laplacian(processedInput, edges, CV_32F, 3);
        edges.convertTo(edges, CV_8U);
    }
    else if (method == "Sobel") {
        cv::Mat sobel_x, sobel_y;
        processedInput.convertTo(processedInput, CV_32F);
        cv::Sobel(processedInput, sobel_x, CV_32F, 1, 0, 3);
        cv::Sobel(processedInput, sobel_y, CV_32F, 0, 1, 3);
        cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, edges);
        // ✅ Normalize before converting to CV_8U (Prevents data loss)
        cv::normalize(edges, edges, 0, 255, cv::NORM_MINMAX);
        edges.convertTo(edges, CV_8U);
    }
    else {
        std::cerr << "ERROR: Invalid method selected! Defaulting to Canny.\n";
        out.convertTo(out, CV_8U);
        cv::Canny(out, edges, 50, 150);
    }
  
    // ✅ Ensure edges exist before further processing
    if (edges.empty()) {
        std::cerr << "WARNING: Edge detection output is empty!" << std::endl;
        out = cv::Mat::zeros(out.size(), CV_8U);
        return;
    }
  
    // ✅ Apply Scharr filter
    //cv::Scharr(edges, scharr_x, CV_8U, 1, 0);
    //cv::Scharr(edges, scharr_y, CV_8U, 0, 1);
    //cv::bitwise_or(scharr_x, scharr_y, scharr_edges); // Faster than addWeighted
    //// ✅ Apply controlled thinning using erosion
    //cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    //cv::erode(scharr_edges, thinned_edges, kernel, cv::Point(-1, -1), 1);
    //cv::dilate(edges, thinned_edges, kernel);  // Much faster than erosion
    //edges.convertTo(thinned_edges, CV_8U); // open if you are close the Scharr filter
    // ✅ Ensure thinning result is valid
    //if (thinned_edges.empty()) {
    //    std::cerr << "WARNING: Thinned edges output is empty! Using edges instead." << std::endl;
    //    thinned_edges = edges.clone();
    //}
    //thinned_edges.convertTo(thinned_edges, CV_8U);
    // ✅ Normalize for better contrast
    //cv::normalize(thinned_edges, thinned_edges, 0, 255, cv::NORM_MINMAX);
  
    // ✅ Blend edges with the original image
    cv::addWeighted(processedInput, 0.5, edges, 0.5, 0, out);
    //cv::bitwise_or(out, thinned_edges, out);

    //cv::Ptr<cv::ORB> orb = cv::ORB::create(100); // Max 500 feature, değiştirilebilir
    //std::vector<cv::KeyPoint> keypoints;
    //cv::Mat descriptors;
//
    //orb->detectAndCompute(out, cv::noArray(), keypoints, descriptors);
//
    //// visualization için: mask'e çiz
    //cv::Mat orb_mask = cv::Mat::zeros(out.size(), CV_8U);
    //for (const auto& kp : keypoints) {
    //    orb_mask.at<uchar>(cv::Point(kp.pt)) = 255;
    //}
    //orb_mask.copyTo(out);
   


  if (prev_event_im.empty() || cv::countNonZero(prev_event_im) == 0 || event_img_.size() != prev_event_im.size()) {
      prev_event_im = cv::Mat::zeros(event_img_.size(), CV_8U); // Siyah bir frame ile başlat
    }
    else {
      //prev_event_im = event_img_;  // Optical flow'da eski frame'in gerçekten değiştiğinden emin ol
      //event_img_.copyTo(prev_event_im);
      event_img_.copyTo(prev_event_im);
      
    }

}*/

// --------------------------------------------- NEW CODE ENDs ------------------------------------------------------------


// -----------------------------------------------------------------------------
std::shared_ptr<NFrame> FrontendBase::createNFrame(
    const std::vector<std::pair<int64_t, ImageBase::Ptr>>& stamped_images)
{
  auto t = timers_[Timer::create_frame].timeScope();

  ++frame_count_;
  NFrame::Ptr nframe =
      states_.makeAndStoreNewNFrame(stamped_images,
                                    FLAGS_imp_detector_max_features_per_frame);
  nframe->setSeq(frame_count_);
  return nframe;
}

//------------------------------------------------------------------------------
std::pair<std::vector<real_t>, uint32_t> FrontendBase::trackFrameKLT()
{
  NFrame::Ptr nframe_k   = states_.nframeK();
  NFrame::Ptr nframe_km1 = states_.nframeKm1();
  NFrame::Ptr nframe_lkf = states_.nframeLkf();

  // Project landmarks in Nframe using klt.
  {
    auto t = timers_[Timer::track_features].timeScope();
    feature_tracker_->trackFeaturesInNFrame(
          states_.T_Bkm1_W(), states_.T_Bk_W(), *nframe_km1, *nframe_k);
  }

  // Outlier rejection using RANSAC.
  std::vector<real_t> disparities_sq;
  uint32_t num_outliers = 0u;
  {
    auto t = timers_[Timer::ransac_relative_pose].timeScope();
    const Transformation T_Bk_Blkf = states_.T_Bk_W() * states_.T_Blkf_W().inverse();

    //! @todo: should we compute the disparity w.r.t the last frame or keyframe?
    std::tie(disparities_sq, num_outliers) =
        feature_tracker_->outlierRemoval(*nframe_lkf, *nframe_k, T_Bk_Blkf);
  }

  // Set last observation in landmarks:
  setLandmarksLastObservationInNFrame(*nframe_k, landmarks_);

  return std::make_pair(disparities_sq, num_outliers);
}

// -----------------------------------------------------------------------------
bool FrontendBase::addImuMeasurementsBetweenKeyframes(
    const ImuStamps& imu_stamps,
    const ImuAccGyrContainer& imu_accgyr)
{
  // Check that there are at least 3 IMU measurements. This is necessary because
  // later we access the third last element, so better have 3 elements, or we
  // will have a neat seg fault.
  if (imu_stamps.size() < 3) {
    LOG(ERROR) << "Less than 3 IMU measurements in vector.";
    return false;
  }

  // TODO: this will only work with one IMU:
  // * One option, do a loop over all the IMUs or
  // * Second option, make the code work only with one IMU,
  // for which all the std::vector<IMUthings> will be unnecessary...
  if (imu_stamps_since_lkf_.size() > 0)
  {
    // Check that last imu stamp since last key frame is equal to
    // the first imu stamp passed to this function.
    //LOG(INFO) << "imu_stamps_since_lkf_ size: " << imu_stamps_since_lkf_.size();
    //LOG(INFO) << "imu_stamps size: " << imu_stamps.size();
    DEBUG_CHECK_EQ(
        imu_stamps_since_lkf_[imu_stamps_since_lkf_.size() - 1],
        imu_stamps[0]
    );

    // TODO: the thing below looks like nonsense to me...
    // ** Detect the case where the first 2 values of the new data are interpolated
    // e.g. t_1 - t_0 < t_2 - t_1. In that case, we want to drop the two interpolated
    // values and only add the actual measurements. **
    //
    // What they really do is:
    // 1) Find the expected period of IMU messages T.
    // 2) If the first two stamps in imu_stamps are separated by less than T,
    // then consider the first measurement an interpolation, hence drop both drop
    // the first stamp in imu_stamps and the last element of imu_stamps_since_lkf
    // (since they are the same according to the DEBUG_CHECK_EQ above).
    // They also do not add the imu_accgyr value corresponding to this stamp to
    // the imu_accgyr_since_lkf_ object, also because it is an interpolated
    // value and not an actual measurement.
    uint64_t n = imu_stamps.size();
    // The -1 here makes the last element in imu_stamps_since_lkf to be overriden by
    // the first (or second, if first is an interpolation) element in imu_stamps.
    // This makes sense as long as the DEBUG_CHECK_EQ above holds.
    uint64_t size_new = imu_stamps_since_lkf_.size() + n - 1;
    uint64_t current_size = imu_stamps_since_lkf_.size();

    // Only do the following if there are more than 1 IMU measurement
    if (imu_stamps.cols() > 1) {
      // **The 2nd and 3rd last elements are never interpolated.**
      // This is actually false when we only have 3 IMU measurements.
      // Below we check whether the fist one is interpolated, and here we say
      // that the 2nd and 3rd (aka first and second) imu measurements are never interpolated...
      // delta_t_actual == delta_t_start if only 3 IMU measurements are present...
      // Therefore we won't drop the first IMU measurement which might be an
      // interpolation.
      int64_t delta_t_actual = imu_stamps_since_lkf_[current_size - 2] -
          imu_stamps_since_lkf_[current_size - 3];
      int64_t delta_t_start = imu_stamps[1] - imu_stamps[0];

      // Check that the time difference between the first two IMU measurements is
      // lower or equal to the actual expected IMU sampling period.
      // Interpolations between IMU measurements should not give higher time differences
      // than the sampling period.
      // This check is to ensure that the next "if" is meaningful.
      LOG(WARNING) << "Number of IMU messages:" << imu_stamps.cols();

      DEBUG_CHECK_LE(delta_t_start, delta_t_actual);
      // **This applies a somewhat arbitrary tolerance on the sampling frequency
      // fluctuations.**
      // Check wether the sampling period of the first measurements is equal
      // to the expected sampling period of the IMU. If it is not, then consider
      // the first IMU measurement an interpolated value rather than a measurement.
      static constexpr int64_t kSamplePeriodTolerance = 100;
      if ((delta_t_actual - delta_t_start) > kSamplePeriodTolerance)
      {
        // This should ensure we remove the interpolated IMU measurement.
        // Since this interpolated value is present as the last element of
        // imu_stamps_since_lkf_ and first measurement of imu_stamps, we have
        // to reduce n and size_new accordingly.
        --n;
        --size_new;
      }
    }

      // **Imu Measurements between frames contain the start and end-times (inclusive)**
      //
      imu_stamps_since_lkf_.conservativeResize(size_new);
      imu_accgyr_since_lkf_.conservativeResize(6, size_new);
      imu_stamps_since_lkf_.tail(n) = imu_stamps.tail(n);
      imu_accgyr_since_lkf_.rightCols(n) = imu_accgyr.rightCols(n);
  }
  else
  {
    imu_stamps_since_lkf_ = imu_stamps;
    imu_accgyr_since_lkf_ = imu_accgyr;
  }
  return true;
}

// -----------------------------------------------------------------------------
void FrontendBase::cleanupInactiveLandmarksFromLastIteration()
{
  landmarks_.cleanupInactiveLandmarks();
  VLOG(40) << landmarks_.typesFormattedString();
}

// -----------------------------------------------------------------------------
void FrontendBase::reset()
{
  landmarks_ = LandmarkTable();
  states_ = NFrameTable();
  frame_count_ = -1;
  attitude_init_count_ = 0;
  stage_ = FrontendStage::AttitudeEstimation;
}

// -----------------------------------------------------------------------------
void FrontendBase::shutdown()
{
  timers_.saveToFile(FLAGS_log_dir, "timings_frontend.yaml");
  stats_.saveToFile(FLAGS_log_dir, "stats_frontend.yaml");
  stopThread();
  visualizer_->stopThread();
}

// -----------------------------------------------------------------------------
VioMotionType FrontendBase::classifyMotion(
    std::vector<real_t>& disparities_sq,
    const uint32_t num_outliers)
{
  if (motion_type_ == VioMotionType::NoMotion)
  {
    return VioMotionType::NoMotion;
  }

  const uint32_t num_inliers = disparities_sq.size();

  // Check if we have enough features:
  if (num_inliers < 15u) // it was 20 
  {
    LOG(WARNING) << "Motion Type: INVALID (tracking " << num_inliers
                 << " features, < 20)";
    return VioMotionType::Invalid;
  }

  // Check that we have X% inliers:
  const real_t inlier_ratio =
      static_cast<real_t>(num_inliers) / (num_inliers + num_outliers);
  if (inlier_ratio < 0.6)
  {
    LOG(WARNING) << "Motion Type: INVALID (inlier ratio = "
                 << inlier_ratio * 100.0 << "%)";
    return VioMotionType::Invalid;
  }

  // Check disparity:
  real_t median_disparity = std::sqrt(median(disparities_sq).first);
  VLOG(10) << "Median Disparity = " << median_disparity;
  if (median_disparity < FLAGS_vio_disparity_median_for_static_motion_classification)
  {
    VLOG(10) << "Motion Type: ROTATION ONLY";
    return VioMotionType::RotationOnly;
  }
  else
  {
    VLOG(10) << "Motion Type: GENERAL";
    return VioMotionType::GeneralMotion;
  }
}

bool FrontendBase::pollBackend(bool block)
{
  // Return false if no updates from backend are available.
  if (!update_states_cb_(block))
    return false;

  // Backend states somehow are not normalized.
  states_.T_Bk_W().getRotation().normalize();

  // Reset integration point using newest result from backend.
  odom_.v_W = states_.v_W(states_.nframeHandleK());

  ImuStamps imu_timestamps;
  ImuAccGyrContainer imu_measurements;

  std::tie(imu_timestamps, imu_measurements) =
    imu_buffer_.getBetweenValuesInterpolated(
      states_.nframeK()->timestamp(),
      odom_.stamp);

  Transformation T_Bkm1_Bk =
    imu_integrator_->integrateImu(
      imu_timestamps,
      imu_measurements,
      states_.T_Bk_W(),
      odom_.v_W);

  odom_.T_W_B = states_.T_Bk_W().inverse() * T_Bkm1_Bk;

  // Attitude is estimated from the IMU in the backend.
  // Hence, continue to Initializing stage when first update has arrived.
  if (stage_ == FrontendStage::AttitudeEstimation)
  {
      stage_ = FrontendStage::Initializing;
  }

  return true;
}

void FrontendBase::addImuData(
    int64_t stamp, const Vector3& acc, const Vector3& gyr, const uint32_t imu_idx)
{
  if (imu_idx != 0)
    return;

  // Add measurement to buffer
  Vector6 acc_gyr;
  acc_gyr.head<3>() = acc;
  acc_gyr.tail<3>() = gyr;
  imu_buffer_.insert(stamp, acc_gyr);
  ++imu_meas_count_;

  real_t dt = nanosecToSecTrunc(stamp - odom_.stamp);
  odom_.stamp = stamp;

  if (stage_ == FrontendStage::Running)
  {
    if (!pollBackend())
    {
      imu_integrator_->propagate(
            odom_.T_W_B.getRotation(), odom_.T_W_B.getPosition(), odom_.v_W,
            acc, gyr, dt);
    }

    odom_.omega_B = gyr - imu_integrator_->getGyrBias();

    visualizer_->publishOdometry(odom_);
  }


}

// -----------------------------------------------------------------------------
bool FrontendBase::limitNumberOfFrames(int max_frames)
{
  if (frame_count_ > max_frames)
  {
    LOG(WARNING) << "This is a demo version of ze_vo that does not allow "
                 << "processing of more than " << max_frames << " frames.";
    if (result_cb_)
    {
      result_cb_(
            0,
            states_.T_Bk_W().getEigenQuaternion().cast<double>(),
            states_.T_Bk_W().getPosition().cast<double>(),
            FrontendStage::Paused, 0u);
    }
    return true;
  }
  else if (frame_count_ == 1)
  {
    LOG(WARNING) << "!!!";
    LOG(WARNING) << "This is a demo version of ze_vo that will not allow "
                 << "processing of more than " << max_frames << " frames.";
    LOG(WARNING) << "!!!";
  }
  return false;
}

// -----------------------------------------------------------------------------
void FrontendBase::startThread()
{
  CHECK(!thread_) << "Thread was already started.";
  thread_.reset(new std::thread(&FrontendBase::frontendLoop, this));
}

// -----------------------------------------------------------------------------
void FrontendBase::stopThread()
{
  if (thread_)
  {
    VLOG(1) << "Interrupt and stop thread.";
    stop_thread_ = true;
    thread_->join();
    thread_.reset();
  }
  VLOG(1) << "Thread stopped and joined.";
}

// -----------------------------------------------------------------------------
void FrontendBase::frontendLoop()
{
  VLOG(1) << "Started frontend thread.";
  while (!stop_thread_)
  {
    FrontendInputData data;
    if (input_queue_.timedRead(data, 100))
    {
      processData(data.stamped_events, data.imu_stamps_vec, data.imu_accgyr_vec);
    }
  }
}

} // namespace ze
