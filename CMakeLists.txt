cmake_minimum_required(VERSION 3.12.2)

project(Lab1 CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT (MSVC_IDE OR XCODE))
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type, one of: Release, Debug" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Configuration types: ${CMAKE_CONFIGURATION_TYPES}")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "A 64-bit OS detected")
else()
    message(STATUS "A 32-bit OS detected")
endif()

if(WIN32)
    set(OpenCV_DIR "D:/_Program Files/opencv/build/x64/vc15/lib")
elseif(APPLE)
    set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "  config: ${OpenCV_DIR}")
message(STATUS "  version: ${OpenCV_VERSION}")
message(STATUS "  libraries: ${OpenCV_LIBS}")
message(STATUS "  include path: ${OpenCV_INCLUDE_DIRS}")

file(GLOB SRC_FILES CONFIGURE_DEPENDS *.cpp *.c)

add_executable(Lab1 ${SRC_FILES})

target_include_directories(Lab1 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Lab1 PRIVATE ${OpenCV_LIBS})