// orb_detector.cpp
#include <imp/features/orb_detector.hpp>
#include <imp/features/opencv_detector_utils.hpp>
#include <opencv2/core.hpp>

namespace ze {

// Basit toCvMat helper'ı
//cv::Mat toCvMat(const ze::Image8uC1& image)
//{
//  return cv::Mat(image.height(), image.width(), CV_8UC1,
//                 const_cast<void*>(static_cast<const void*>(image.data())));
//}
//
// Basit convertCvKeypoints helper'ı
//void convertCvKeypoints(const std::vector<cv::KeyPoint>& cv_kps, ze::KeypointsWrapper* kp_out)
//{
//  size_t n = cv_kps.size();
//  kp_out->px.resize(Eigen::NoChange, n);
//  for (size_t i = 0; i < n; ++i)
//  {
//    kp_out->px(0, i) = cv_kps[i].pt.x;
//    kp_out->px(1, i) = cv_kps[i].pt.y;
//  }
//  kp_out->num_detected = static_cast<int>(n);
//}

OrbDetector::OrbDetector(const cv::Ptr<cv::ORB>& orb,
  const Size2u& image_size)
: AbstractDetector(image_size, DetectorType::Orb)
, orb_(orb)
{}

uint32_t OrbDetector::detect(const ImagePyramid8uC1& pyr, KeypointsWrapper& keypoints)
{
  const cv::Mat img = toCvMat(pyr[0]);

  std::vector<cv::KeyPoint> keypoints_cv;
  if (mask_)
  {
  const cv::Mat cv_mask = toCvMat(*mask_);
  orb_->detect(img, keypoints_cv, cv_mask);
  }
  else
  {
  orb_->detect(img, keypoints_cv);
  }

  //for (const auto& kp : keypoints_cv)
  //{
  //  keypoints.addKeypoint(
  //    kp.pt.x, kp.pt.y, kp.response, kp.octave, kp.angle,
  //    static_cast<uint8_t>(DetectorType::Orb));
  //  
  //}

  //return static_cast<uint32_t>(keypoints_cv.size());
  //uint32_t added = 0;
  //for (const auto& kp : keypoints_cv)
  //{
  //  if (keypoints.addKeypoint(
  //        kp.pt.x, kp.pt.y, kp.response, kp.octave, kp.angle,
  //        static_cast<uint8_t>(DetectorType::Orb)))
  //  {
  //    ++added;
  //  }
  //}
//
  //return added;

  // Grid parametreleri
  const int grid_rows = 20;  //20; // örnek: 8x8 grid
  const int grid_cols = 20;  //20;
  const int cell_width = img.cols / grid_cols;
  const int cell_height = img.rows / grid_rows;

  std::vector<cv::KeyPoint> selected_keypoints;
  selected_keypoints.reserve(grid_rows * grid_cols);

  // Her hücrede en iyi keypoint’i bul
  for (int row = 0; row < grid_rows; ++row)
  {
    for (int col = 0; col < grid_cols; ++col)
    {
      const int x0 = col * cell_width;
      const int y0 = row * cell_height;
      const int x1 = (col + 1) * cell_width;
      const int y1 = (row + 1) * cell_height;

      float best_response = -1.f;
      cv::KeyPoint best_kp;

      for (const auto& kp : keypoints_cv)
      {
        if (kp.pt.x >= x0 && kp.pt.x < x1 &&
            kp.pt.y >= y0 && kp.pt.y < y1)
        {
          if (kp.response > best_response)
          {
            best_response = kp.response;
            best_kp = kp;
          }
        }
      }

      if (best_response > 0.f)
      {
        selected_keypoints.push_back(best_kp);
      }
    }
  }

  // En iyi keypoint’leri addKeypoint() ile ekle
  uint32_t added = 0;
  for (const auto& kp : selected_keypoints)
  {
    if (keypoints.addKeypoint(
          kp.pt.x, kp.pt.y, kp.response, kp.octave, kp.angle,
          static_cast<uint8_t>(DetectorType::Orb)))
    {
      ++added;
    }
  }

  //LOG(INFO) << "[ORB] Selected " << selected_keypoints.size()
  //          << " grid keypoints → added " << added;

  return added;

}

void OrbDetector::setMask(const Image8uC1::ConstPtr& mask)
{
 mask_ = mask;
}

cv::Mat toCvMat(const Image8uC1& image)
{
 return cv::Mat(image.height(), image.width(), CV_8UC1, const_cast<void*>(static_cast<const void*>(image.data())));
}

} // namespace ze
