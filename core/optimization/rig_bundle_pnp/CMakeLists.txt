cmake_minimum_required(VERSION 3.5)
project(RigBundle)

# Find required packages
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(cnpy STATIC ${CMAKE_CURRENT_SOURCE_DIR}/include/cnpy.cpp)
# Add executables
# add_executable(generate_data generate_data.cpp)
add_executable(optimize_rig optimize_rig.cpp)

# Link libraries
target_link_libraries(optimize_rig ${CERES_LIBRARIES})

# Set C++11 or higher standard if needed
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Make sure we have nlohmann/json
include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Link nlohmann/json to both executables
target_link_libraries(optimize_rig nlohmann_json::nlohmann_json cnpy)

# Output binaries to bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
