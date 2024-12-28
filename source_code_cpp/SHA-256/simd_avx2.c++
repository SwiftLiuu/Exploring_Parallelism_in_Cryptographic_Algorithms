#include <stdio.h>
#include <stdlib.h>
#include<omp.h>
#include<Windows.h>
#include<fstream>
#include<immintrin.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <sys/stat.h>  // For fstat
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <vector>
#include <cstdlib>
#define NUM 2
#pragma warning(disable:4996);
#define SHA256_ROTL(a,b) (_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(a,32-b), _mm256_set1_epi32((0x7fffffff>>(31-b)))),_mm256_slli_epi32(a,b)))
#define SHA256_SR(a,b) (_mm256_and_si256((_mm256_srli_epi32(a,b)), _mm256_set1_epi32((0x7fffffff>>(b-1)))))
#define SHA256_Ch(x,y,z) (_mm256_xor_si256(_mm256_and_si256(x,y),_mm256_and_si256(_mm256_xor_si256(x,_mm256_set1_epi32(0xffffffff)),z)))
#define SHA256_Maj(x,y,z) (_mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(x,y),_mm256_and_si256(x,z)),_mm256_and_si256(y,z)))
#define SHA256_E0(x) _mm256_xor_si256(_mm256_xor_si256(SHA256_ROTL(x,30),SHA256_ROTL(x,19)),SHA256_ROTL(x,10))
#define SHA256_E1(x) _mm256_xor_si256(_mm256_xor_si256(SHA256_ROTL(x,26),SHA256_ROTL(x,21)),SHA256_ROTL(x,7))
#define SHA256_O0(x) _mm256_xor_si256(_mm256_xor_si256(SHA256_ROTL(x,25),SHA256_ROTL(x,14)),SHA256_SR(x,3))
#define SHA256_O1(x)  _mm256_xor_si256(_mm256_xor_si256(SHA256_ROTL(x,15),SHA256_ROTL(x,13)),SHA256_SR(x,10))
using namespace std;
char text[NUM + 1];
class ThreadPool {
public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                        if (stop_ && tasks_.empty())
                            return;

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    task();

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        --active_threads_;
                    }

                    completed_cv_.notify_one();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }

        cv_.notify_all();

        for (auto& thread : threads_)
            thread.join();
    }

    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::move(task));
            ++active_threads_;
        }
        cv_.notify_one();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        completed_cv_.wait(lock, [this] { return tasks_.empty() && active_threads_ == 0; });
    }

private:
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::condition_variable completed_cv_;
    bool stop_ = false;
    size_t active_threads_ = 0; // Tracks the number of actively executing tasks
};
int get_block_count(int length) {
    int l = length + ((length % 64 >= 56) ? (128 - length % 64) : (64 - length % 64));
    return l / 64;
}

int init(char* ss[8], long lengths[8], long* M[8]) {
    for (int k = 0; k < 8; k++) {
        int length = lengths[k];
        int l = length + ((length % 64 >= 56) ? (128 - length % 64) : (64 - length % 64));
        int num_blocks = l / 64;

        char* buffer = (char*)malloc(l);
        if (!buffer) return 0;

        for (int i = 0; i < length; i++) {
            buffer[i ^ 3] = ss[k][i];
        }

        buffer[length ^ 3] = 0x80;
        for (int i = length + 1; i < l - 8; i++) {
            buffer[i ^ 3] = 0x00;
        }

        unsigned long long bit_length = (unsigned long long)length * 8;
        for (int i = 0; i < 8; i++) {
            buffer[(l - 1 - i) ^ 3] = (bit_length >> (i * 8)) & 0xFF;
        }

        M[k] = (long*)malloc(num_blocks * 16 * sizeof(long));
        if (!M[k]) {
            free(buffer);
            return 0;
        }

        char* buffer_ptr = buffer;
        for (int i = 0; i < num_blocks; i++, buffer_ptr += 64) {
            for (int j = 0; j < 16; j++) {
                M[k][i * 16 + j] = ((long*)buffer_ptr)[j];
            }
        }

        free(buffer);
    }

    return 1;
}

void StrSHA256(long* M[8], int num_blocks, char* sha2560, char* sha2561, char* sha2562, char* sha2563,char* sha2564, char* sha2565, char* sha2566, char* sha2567) {
    __m256i A, B, C, D, E, F, G, H, T1, T2;
    __m256i H0 = _mm256_set1_epi32(0x6a09e667);
    __m256i H1 = _mm256_set1_epi32(0xbb67ae85);
    __m256i H2 = _mm256_set1_epi32(0x3c6ef372);
    __m256i H3 = _mm256_set1_epi32(0xa54ff53a);
    __m256i H4 = _mm256_set1_epi32(0x510e527f);
    __m256i H5 = _mm256_set1_epi32(0x9b05688c);
    __m256i H6 = _mm256_set1_epi32(0x1f83d9ab);
    __m256i H7 = _mm256_set1_epi32(0x5be0cd19);
    unsigned long int Ki[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    __m256i K[64];
    for (int i = 0; i < 64; K[i] = _mm256_set1_epi32(Ki[i]), i++);

    __m256i W[64];
    
    for (int b = 0; b < num_blocks; b++) {
        for (int c = 0; c < 16; W[c] = _mm256_set_epi32(M[0][b * 16 + c], M[1][b * 16 + c], M[2][b * 16 + c], M[3][b * 16 + c], M[4][b * 16 + c], M[5][b * 16 + c], M[6][b * 16 + c], M[7][b * 16 + c]), c++);

        for (int i = 16; i < 64; i++) {
            W[i] = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(SHA256_O1(W[i - 2]), W[i - 7]), SHA256_O0(W[i - 15])), W[i - 16]);
        }

        A = H0, B = H1, C = H2, D = H3, E = H4, F = H5, G = H6, H = H7;
        for (int i = 0; i < 64; i++) {
            T1 = _mm256_add_epi32(H, _mm256_add_epi32(SHA256_E1(E), _mm256_add_epi32(SHA256_Ch(E, F, G), _mm256_add_epi32(K[i], W[i]))));
            T2 = _mm256_add_epi32(SHA256_E0(A), SHA256_Maj(A, B, C));
            H = G, G = F, F = E, E = _mm256_add_epi32(D, T1), D = C, C = B, B = A, A = _mm256_add_epi32(T1, T2);
        }

        H0 = _mm256_add_epi32(H0, A), H1 = _mm256_add_epi32(H1, B), H2 = _mm256_add_epi32(H2, C), H3 = _mm256_add_epi32(H3, D);
        H4 = _mm256_add_epi32(H4, E), H5 = _mm256_add_epi32(H5, F), H6 = _mm256_add_epi32(H6, G), H7 = _mm256_add_epi32(H7, H);
    }

    long add_out[8][8];
    _mm256_storeu_si256((__m256i*)add_out[0], H0);
    _mm256_storeu_si256((__m256i*)add_out[1], H1);
    _mm256_storeu_si256((__m256i*)add_out[2], H2);
    _mm256_storeu_si256((__m256i*)add_out[3], H3);
    _mm256_storeu_si256((__m256i*)add_out[4], H4);
    _mm256_storeu_si256((__m256i*)add_out[5], H5);
    _mm256_storeu_si256((__m256i*)add_out[6], H6);
    _mm256_storeu_si256((__m256i*)add_out[7], H7);

    sprintf(sha2560, "%08X%08X%08X%08X%08X%08X%08X%08X", add_out[0][0], add_out[1][0], add_out[2][0], add_out[3][0], add_out[4][0], add_out[5][0], add_out[6][0], add_out[7][0]);
    sprintf(sha2561, "%08X%08X%08X%08X%08X%08X%08X%08X", add_out[0][1], add_out[1][1], add_out[2][1], add_out[3][1], add_out[4][1], add_out[5][1], add_out[6][1], add_out[7][1]);
    sprintf(sha2562, "%08X%08X%08X%08X%08X%08X%08X%08X", add_out[0][2], add_out[1][2], add_out[2][2], add_out[3][2], add_out[4][2], add_out[5][2], add_out[6][2], add_out[7][2]);
    sprintf(sha2563, "%08X%08X%08X%08X%08X%08X%08X%08X", add_out[0][3], add_out[1][3], add_out[2][3], add_out[3][3], add_out[4][3], add_out[5][3], add_out[6][3], add_out[7][3]);
    sprintf(sha2564, "%08X%08X%08X%08X%08X%08X%08X%08X", add_out[0][4], add_out[1][4], add_out[2][4], add_out[3][4], add_out[4][4], add_out[5][4], add_out[6][4], add_out[7][4]);
    sprintf(sha2565, "%08X%08X%08X%08X%08X%08X%08X%08X", add_out[0][5], add_out[1][5], add_out[2][5], add_out[3][5], add_out[4][5], add_out[5][5], add_out[6][5], add_out[7][5]);
    sprintf(sha2566, "%08X%08X%08X%08X%08X%08X%08X%08X", add_out[0][6], add_out[1][6], add_out[2][6], add_out[3][6], add_out[4][6], add_out[5][6], add_out[6][6], add_out[7][6]);
    sprintf(sha2567, "%08X%08X%08X%08X%08X%08X%08X%08X", add_out[0][7], add_out[1][7], add_out[2][7], add_out[3][7], add_out[4][7], add_out[5][7], add_out[6][7], add_out[7][7]);
}

void StrSHA256_serial(long* M, int num_blocks, char* sha256) {
    unsigned int A, B, C, D, E, F, G, H, T1, T2;
    unsigned int H0 = 0x6a09e667;
    unsigned int H1 = 0xbb67ae85;
    unsigned int H2 = 0x3c6ef372;
    unsigned int H3 = 0xa54ff53a;
    unsigned int H4 = 0x510e527f;
    unsigned int H5 = 0x9b05688c;
    unsigned int H6 = 0x1f83d9ab;
    unsigned int H7 = 0x5be0cd19;
    unsigned long int Ki[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };

    unsigned int W[64];

    for (int b = 0; b < num_blocks; b++) {
        // Load initial 16 words for each block
        for (int c = 0; c < 16; c++) {
            W[c] = (unsigned int)M[b * 16 + c];
        }

        // Prepare the message schedule
        for (int i = 16; i < 64; i++) {
            unsigned int s0 = (W[i - 15] >> 7 | W[i - 15] << (32 - 7)) ^ (W[i - 15] >> 18 | W[i - 15] << (32 - 18)) ^ (W[i - 15] >> 3);
            unsigned int s1 = (W[i - 2] >> 17 | W[i - 2] << (32 - 17)) ^ (W[i - 2] >> 19 | W[i - 2] << (32 - 19)) ^ (W[i - 2] >> 10);
            W[i] = W[i - 16] + s0 + W[i - 7] + s1;
        }

        // Initialize working variables
        A = H0;
        B = H1;
        C = H2;
        D = H3;
        E = H4;
        F = H5;
        G = H6;
        H = H7;

        // Main compression loop
        for (int i = 0; i < 64; i++) {
            unsigned int S1 = (E >> 6 | E << (32 - 6)) ^ (E >> 11 | E << (32 - 11)) ^ (E >> 25 | E << (32 - 25));
            unsigned int ch = (E & F) ^ ((~E) & G);
            T1 = H + S1 + ch + Ki[i] + W[i];
            unsigned int S0 = (A >> 2 | A << (32 - 2)) ^ (A >> 13 | A << (32 - 13)) ^ (A >> 22 | A << (32 - 22));
            unsigned int maj = (A & B) ^ (A & C) ^ (B & C);
            T2 = S0 + maj;

            H = G;
            G = F;
            F = E;
            E = D + T1;
            D = C;
            C = B;
            B = A;
            A = T1 + T2;
        }

        // Update hash values
        H0 += A;
        H1 += B;
        H2 += C;
        H3 += D;
        H4 += E;
        H5 += F;
        H6 += G;
        H7 += H;
    }
    
    // Produce the final hash value (big-endian)
    sprintf(sha256, "%08X%08X%08X%08X%08X%08X%08X%08X", H0, H1, H2, H3, H4, H5, H6, H7);
}
void read_files_into_buffers(const char* filenames[8], char* buffers[8], long lengths[8]) {
    // Read each file into a buffer
    for (int i = 0; i < 8; i++) {
        FILE* file = fopen(filenames[i], "rb");
        if (!file) {
            perror("Failed to open file");
            exit(1);
        }

        // Get the file descriptor and use fstat to determine the file size
        int fd = fileno(file);
        struct stat file_stat;
        if (fstat(fd, &file_stat) != 0) {
            perror("Failed to determine file size");
            fclose(file);
            exit(1);
        }

        lengths[i] = file_stat.st_size;

        // Allocate buffer and read file content
        buffers[i] = (char*)malloc(lengths[i]);
        if (!buffers[i]) {
            perror("Failed to allocate memory");
            fclose(file);
            exit(1);
        }

        size_t bytesRead = fread(buffers[i], 1, lengths[i], file);
        if (bytesRead != lengths[i]) {
            perror("Failed to read the entire file");
            free(buffers[i]);
            fclose(file);
            exit(1);
        }

        fclose(file);
    }
}
char* random_str(int length) {
    static const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    char* result = (char*)malloc(length + 1);
    if (!result) return NULL;

    for (int i = 0; i < length; ++i) {
        result[i] = charset[rand() % (sizeof(charset) - 1)];
    }
    result[length] = '\0';

    return result;
}
int main() {
    // char* ss[4] = { "aaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccddd","aaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccddd","aaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccddd","aaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccdddaaaabbbbccccddd" };
    // char* ss[4] = { "abc","aaaabbbbc","aaaabbbbccccddd","aaaabbbbccccddd" };
    // const char* filenames[4] = { "E:/PP/project/test1.txt", "E:/PP/project/test2.txt", "E:/PP/project/test3.txt", "E:/PP/project/test4.txt" };
    // const char* filenames[4] = { "E:/PP/project/50MB.bin", "E:/PP/project/50MB (1).bin", "E:/PP/project/50MB (2).bin", "E:/PP/project/50MB (3).bin" };
    // const char* filenames[4] = { "E:/PP/project/2GB.bin", "E:/PP/project/2GB.bin", "E:/PP/project/2GB.bin", "E:/PP/project/2GB.bin" };
    // const char* filenames[4] = { "E:/PP/project/test4.txt", "E:/PP/project/test4.txt", "E:/PP/project/test4.txt", "E:/PP/project/test4.txt" };
    // const char* filenames[4] = { "E:/PP/project/assg1.exe", "E:/PP/project/assg1.exe", "E:/PP/project/assg1.exe", "E:/PP/project/assg1.exe" };
//     const char* filenames[8] = {
//     "E:/PP/project/test4.txt",
//     "E:/PP/project/test4.txt",
//     "E:/PP/project/test4.txt",
//     "E:/PP/project/test4.txt",
//     "E:/PP/project/test4.txt",
//     "E:/PP/project/test4.txt",
//     "E:/PP/project/test4.txt",
//     "E:/PP/project/test4.txt"
// };
    const int num_strings = 1024;
    const int num_blocks = 200; 
    char* ss[num_strings];
    for (int i = 0; i < num_strings ; i++) {
         ss[i] = random_str(30+(num_blocks-1)*64);
    }
    // printf(ss[0]);
    // char* ss[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    long lengths[8];
    // read_files_into_buffers(filenames,ss,lengths);

    long* M[8];
    long* M_all[num_strings/8][8];
    int num_threads =16;
    ThreadPool pool(num_threads);
    
    char sha2560_par[num_strings][65];
    char sha2561_par[num_strings][65];
    char sha2562_par[num_strings][65];
    char sha2563_par[num_strings][65];
    char sha2564_par[num_strings][65];
    char sha2565_par[num_strings][65];
    char sha2566_par[num_strings][65];
    char sha2567_par[num_strings][65];
    int max_blocks=0;
    char sha256_ser[8][65];
    for (int i = 0; i < 8; i++) {
        lengths[i] = strlen(ss[i]);
        // printf("Length of ss[%d] = %d\n", i, lengths[i]);
    }
    for (int i=0;i<num_strings/8;i++){
        if (init(ss, lengths, M)) {
            for (int k = 0; k < 8; k++) {
                int blocks=get_block_count(lengths[k]);
                if (max_blocks<blocks){
                    max_blocks=blocks;
                }
                // int output_length=get_block_count(lengths[k])*16;
                // printf("Block %d:\n", k);
                // for (int i = 0; i < output_length; i++) {
                //     printf("M[%d][%d] = %lx\n", k, i, M[k][i]);
                // }
            }
            for (int j = 0; j < 8; j++) {
                M_all[i][j] = M[j]; // Assign each pointer from M to M_all[i][j]
            }
        } else {
            printf("Initialization failed.\n");
        }
    }
    
    printf("%d\n",max_blocks);
    char sha2560[65],sha2561[65],sha2562[65],sha2563[65],sha2564[65],sha2565[65],sha2566[65],sha2567[65];
    for (int i = 0; i < 1; i++)
    {
        LARGE_INTEGER  num;
        long long start, end, freq;
        int test_times=20;
        QueryPerformanceFrequency(&num);
        freq = num.QuadPart;
        QueryPerformanceCounter(&num);
        start = num.QuadPart;
        for (int i = 0; i < test_times; i++){

        
            for (int j = 0; j < num_strings/8; j++)
            {
                pool.enqueue([&, j] {
                // Compute hashes for a specific string
                StrSHA256(M_all[j], max_blocks, sha2560_par[j], sha2561_par[j], sha2562_par[j], sha2563_par[j],sha2564_par[j], sha2565_par[j], sha2566_par[j], sha2567_par[j]);
            });
            }
            pool.wait();
        }
        QueryPerformanceCounter(&num);
        end = num.QuadPart;
        printf("time=%d us\n", (end - start) * 1000000 /(test_times*freq));
         puts(sha2567_par[0]);

        // QueryPerformanceCounter(&num);
        // start = num.QuadPart;
        // for (int j = 0; j < num_strings*test_times/8; j++)
        // {
        //     for (int k = 0; k < 8; k++) {
        //         StrSHA256_serial(M[k],max_blocks,sha256_ser[k]);
        //     }
        // }
        // QueryPerformanceCounter(&num);
        // end = num.QuadPart;
        // printf("time=%d us\n", (end - start) * 1000000 /(test_times*freq));
        // puts(sha256_ser[0]);
    }
    return 0;
}
