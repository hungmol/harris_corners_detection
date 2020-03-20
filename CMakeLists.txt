cmake_minimum_required(VERSION 3.10)

project(HarrisCorners)
set(CMAKE_CXX_STANDARD 17)


set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/HarrisCorners.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/HarrisCorners.h
    )

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} -lstdc++fs)
    