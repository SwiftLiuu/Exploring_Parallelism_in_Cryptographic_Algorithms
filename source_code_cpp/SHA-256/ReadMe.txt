//author: Zihao Gong 1005036916
//The code can only run on Windows since the time function Im using is based on Windows high accuracy time.

//To build file,use command 
simd.c++:  g++ -mavx -O2 -o output simd.c++
simd_avx2.c++:   g++ -mavx2 -O2 -o output simd_avx2.c++
cuda_lots_output.cu: nvcc -O2 -o output cuda_lots_output.cu
failure attemp: g++  -O2  thread_pool.c++ -o thread_pool

//To run output,
./output

//To change the input length, change the variable num_blocks, which means the number of 512bits blocks.
//To change the input number, change the variable num_strings.