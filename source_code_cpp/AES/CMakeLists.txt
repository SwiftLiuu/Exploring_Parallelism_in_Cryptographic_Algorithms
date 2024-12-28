cmake_minimum_required(VERSION 3.25.1)

set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)

project(AES_IMP LANGUAGES CXX CUDA)

set(EXE AES_IMP)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "-O3 -arch=compute_86")
set(CMAKE_CXX_FLAGS "-Wall -O3")


set(SOURCES
        serial/AESSerial.cpp
        serial/AESSerial.h
        serial/CONSTANTS.h
        serial/DebugUtil.h
        thread/AESThreaded.cpp
        thread/AESThreaded.h
        thread/AESOpenMP.cpp
        thread/AESOpenMP.h
        Utils/sampler.h
)

set(CUDA_SOURCES
        main.cu
        cu/AEScu.cu
        cu/AEScu.h
        Utils/Generator.h
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -MD -MP")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -MD -MP")

add_executable(${EXE} ${SOURCES} ${CUDA_SOURCES})

# Link CUDA runtime library
target_link_libraries(${EXE} PRIVATE cudart)

# Set include and library directories
target_include_directories(${EXE} PRIVATE ${INCDIRS})
target_link_directories(${EXE} PRIVATE ${LIBDIRS})
target_link_libraries(${EXE} PUBLIC OpenMP::OpenMP_CXX)