cmake_minimum_required(VERSION 3.10.0)
project(m VERSION 0.1.0 LANGUAGES C CXX)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(m main.cpp)
include_directories(${CERES_INCLUDE_DIRS})
target_link_libraries(m PRIVATE ${CERES_LIBRARIES} Eigen3::Eigen)
target_link_libraries(m PRIVATE ${OpenCV_LIBS})
target_include_directories(m PRIVATE ${OpenCV_INCLUDE_DIRS})

