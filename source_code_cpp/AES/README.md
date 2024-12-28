1. install cmake
2. mkdir cmake-build-release
3. cd cmake-build-release
4. cmake -DCMAKE_BUILD_TYPE=Release ..
5. cmake --build .

./AES_IMP <input multiplier> <thread for cpu mode> <coalesce factor> <thread per cuda block multiplier>