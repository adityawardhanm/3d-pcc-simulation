# CMakeLists.txt

# Note : This Cmake has been made and tested for C++ 20 and CUDA 8.9 (nvcc 12.6)
#        If you are using a different version of CUDA and GPU architecture, please change the 89 mentions accordingly.

cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
project(CudaKinematics LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_CUDA_ARCHITECTURES 89) # Targeting NVIDIA Ada Lovelace architecture (compute capability 8.9)

set(KINEMATICS_DIR ${CMAKE_SOURCE_DIR}/src/cuda)

# ===== fk.so =====
add_library(fk SHARED ${KINEMATICS_DIR}/fk.cu
)
set_target_properties(fk 
    PROPERTIES PREFIX ""        # No 'lib' prefix
    POSITION_INDEPENDENT_CODE ON    
)
target_compile_options(fk PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -O3 
    -gencode=arch=compute_89,code=sm_89
    -gencode=arch=compute_89,code=compute_89    
        
    --maxrregcount=0            # unlimited registers
    -Xptxas -v                  # verbose ptxas output
    -Xptxas -O3                 # optimize for speed

    -lineinfo                   # include line number info for profiling and debugging 
    >
)

# -Xptxas -warn-on-spills  - try to enable this later for debugging register spills