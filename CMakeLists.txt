cmake_minimum_required(VERSION 3.20)

project(TrafficSurveillanceSystem)

set(CMAKE_CXX_STANDARD 17)

# Set VCPKG toolchain if not passed externally
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
endif()

# Set include and source directories
include_directories(include)

file(GLOB_RECURSE SOURCES
    "src/*.cpp"
)

add_executable(MyAwesomeApp ${SOURCES})

# Link with vcpkg packages
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(MyAwesomeApp
    PRIVATE
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    Boost::boost
)
