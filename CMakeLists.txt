cmake_minimum_required(VERSION 3.16)
project(amd-ml-operators LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

add_executable(flash_attention flash_attention.hip)
target_link_libraries(flash_attention hip::device)
