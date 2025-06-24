# 要求CMake的最低版本为3.20
cmake_minimum_required(VERSION 3.20)

# 设置项目名称和支持的语言，这里的项目名称可以根据实际情况修改，语言列表包括C、C++和CUDA
project(cuda-cplusplus-programming-guide LANGUAGES C CXX CUDA)

# 将自定义 CMake 模块路径添加到搜索路径中，便于后续 find_package 调用
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

# 查找并加载 CUDA Toolkit 依赖（必须存在）
find_package(CUDAToolkit REQUIRED)

# 编译标准设置
# 启用位置无关代码（PIC），通常用于生成共享库
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# 强制要求 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 强制要求 CUDA 标准为 CUDA 17（对应 CUDA 11.0+）
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

"""
指定目标 GPU 架构（计算能力版本），覆盖从 Maxwell 到 Ampere 的多代显卡：
 - 50/52: Maxwell (GTX 9xx)
 - 60/61: Pascal (GTX 10xx)
 - 70/72/75: Volta/Turing (Titan V, RTX 20xx)
 - 80/86/87/89/90: Ampere (RTX 30xx, A100)
 - 100/101/120: Hopper (H100)
"""
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 72 75 80 86 87 89 90 100 101 120)

# 设置 CUDA 编译器选项
# -Wno-deprecated-gpu-targets: 忽略已弃用的 GPU 目标警告
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
# 设置 CUDA 调试选项
# ENABLE_CUDA_DEBUG: 是否启用 CUDA 调试模式，默认为 OFF
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # 启用 cuda-gdb（可能会严重影响某些目标的性能）
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # 为调试工具的所有构建文件添加行信息（-G 选项独有）
endif()
# 启用扩展 lambda 支持（允许在 __device__ 代码中使用复杂 lambda）
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# 定义创建 CUDA 可执行文件的函数
function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_compile_options(${target_name} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    )
    target_compile_features(${target_name} PRIVATE 
        cxx_std_17 
        cuda_std_17
    )
    set_target_properties(${target_name} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

# 进入各个子目录继续构建实际示例代码
add_subdirectory(chapter_05)