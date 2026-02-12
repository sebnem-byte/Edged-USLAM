// orb_detector.hpp
#pragma once

#include <ze/common/types.hpp>
#include <imp/imgproc/image_pyramid.hpp>
#include <imp/features/keypoints_wrapper.hpp>
#include <ze/common/macros.hpp>
#include <imp/features/feature_detector.hpp>
#include <opencv2/features2d.hpp>

namespace ze {

class OrbDetector : public AbstractDetector
{
public:
  ZE_POINTER_TYPEDEFS(OrbDetector);

  OrbDetector(const cv::Ptr<cv::ORB>& orb,
              const Size2u& image_size);

  uint32_t detect(const ImagePyramid8uC1& pyr,
                  KeypointsWrapper& keypoints) override;

  void reset() override {}

  void setMask(const Image8uC1::ConstPtr& mask) override;

private:
  cv::Ptr<cv::ORB> orb_;
  Image8uC1::ConstPtr mask_ = nullptr;
};

// Basit yardımcı: ze::Image → cv::Mat
cv::Mat toCvMat(const Image8uC1& image);

} // namespace ze
