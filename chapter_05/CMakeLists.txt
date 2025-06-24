# 要求CMake的最低版本为3.20
cmake_minimum_required(VERSION 3.20)

# 设置项目名称和支持的语言，这里的项目名称可以根据实际情况修改，语言列表包括C、C++和CUDA
project(chapter_05 LANGUAGES C CXX CUDA)

# 包含公共目录和库
include_directories(../common)

# 编译源代码为可执行程序
# 获取所有 .cu 文件
file(GLOB CUDA_SOURCES "*.cu")
# 循环处理每个 CUDA 源文件，这里假设每个 .cu 文件对应一个可执行文件
# 如果需要将多个 .cu 文件编译为一个可执行文件，可以调整逻辑
foreach(source_file ${CUDA_SOURCES})
    # 从文件名获取目标名（去掉扩展名）
    get_filename_component(target_name ${source_file} NAME_WE)
    # 创建 CUDA 可执行文件
    add_cuda_executable(${target_name} ${source_file})
endforeach()
