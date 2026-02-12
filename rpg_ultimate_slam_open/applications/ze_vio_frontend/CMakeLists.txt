project(ze_vio_frontend)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

catkin_python_setup()

include(ze_setup)

include_directories(/usr/include/sophus)

#############
# LIBRARIES #
#############
set(HEADERS
  include/ze/vio_frontend/feature_initializer.hpp
  include/ze/vio_frontend/feature_tracker.hpp
  include/ze/vio_frontend/frontend_api.hpp
  include/ze/vio_frontend/frontend_base.hpp
  include/ze/vio_frontend/frontend_gflags.hpp
  include/ze/vio_frontend/frontend_vio.hpp
  include/ze/vio_frontend/keyframe_selection.hpp
  include/ze/vio_frontend/landmarks_reprojector.hpp
  include/ze/vio_frontend/refinement.hpp
  include/ze/vio_frontend/stereo_matcher.hpp
  include/ze/vio_frontend/track_extractor.hpp
  )

set(SOURCES
  src/feature_initializer.cpp
  src/feature_tracker.cpp
  src/frontend_api.cpp
  src/frontend_base.cpp
  src/frontend_gflags.cpp
  src/frontend_logging.cpp
  src/frontend_vio.cpp
  src/keyframe_selection.cpp
  src/landmarks_reprojector.cpp
  src/refinement.cpp
  src/stereo_matcher.cpp
  src/track_extractor.cpp
  src/se3.cpp
  )

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

include_directories(
  ${PROJECT_SOURCE_DIR}/../ze_ceres_backend/include
)

##########
# EXPORT #
##########
cs_install()
cs_export()
