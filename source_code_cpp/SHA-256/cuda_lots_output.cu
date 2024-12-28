#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <string.h>
#include<Windows.h>
#include <condition_variable>
#include <functional>
#include <vector>
#include <time.h>
#define SHA256_ROTL(a, b) (((a) >> (32 - (b))) | ((a) << (b)))
#define SHA256_SR(a, b) ((a) >> (b))
#define SHA256_Ch(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define SHA256_Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SHA256_E0(x) (SHA256_ROTL((x), 30) ^ SHA256_ROTL((x), 19) ^ SHA256_ROTL((x), 10))
#define SHA256_E1(x) (SHA256_ROTL((x), 26) ^ SHA256_ROTL((x), 21) ^ SHA256_ROTL((x), 7))
#define SHA256_O0(x) (SHA256_ROTL((x), 25) ^ SHA256_ROTL((x), 14) ^ SHA256_SR((x), 3))
#define SHA256_O1(x) (SHA256_ROTL((x), 15) ^ SHA256_ROTL((x), 13) ^ SHA256_SR((x), 10))

__device__ void StrSHA256(unsigned int* M, int num_blocks, char* sha256) {
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
        for (int c = 0; c < 16; c++) {
            W[c] = M[b * 16 + c];
        }

        for (int i = 16; i < 64; i++) {
            unsigned int s0 = (W[i - 15] >> 7 | W[i - 15] << (32 - 7)) ^ (W[i - 15] >> 18 | W[i - 15] << (32 - 18)) ^ (W[i - 15] >> 3);
            unsigned int s1 = (W[i - 2] >> 17 | W[i - 2] << (32 - 17)) ^ (W[i - 2] >> 19 | W[i - 2] << (32 - 19)) ^ (W[i - 2] >> 10);
            W[i] = W[i - 16] + s0 + W[i - 7] + s1;
        }

        A = H0;
        B = H1;
        C = H2;
        D = H3;
        E = H4;
        F = H5;
        G = H6;
        H = H7;

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

        H0 += A;
        H1 += B;
        H2 += C;
        H3 += D;
        H4 += E;
        H5 += F;
        H6 += G;
        H7 += H;
    }

    const char hex_chars[] = "0123456789abcdef";
    unsigned int hash_values[8] = {H0, H1, H2, H3, H4, H5, H6, H7};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            unsigned char byte = (hash_values[i] >> (24 - j * 8)) & 0xFF;
            sha256[i * 8 + j * 2] = hex_chars[(byte >> 4) & 0xF];
            sha256[i * 8 + j * 2 + 1] = hex_chars[byte & 0xF];
        }
    }
    sha256[64] = '\0';
}

__global__ void sha256_kernel(unsigned int* M, int num_blocks, char* sha256_outputs, int num_strings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_strings) {
        StrSHA256(&M[idx * num_blocks * 16], num_blocks, &sha256_outputs[idx * 65]);
    }
}
int init(char* ss, long length, unsigned int** M) {
    int l = length + ((length % 64 >= 56) ? (128 - length % 64) : (64 - length % 64));
    int num_blocks = l / 64;

    // Allocate memory for padded data
    char* buffer = (char*)malloc(l);
    if (!buffer) return 0;

    // Copy input data and adjust byte order
    for (int i = 0; i < length; i++) {
        buffer[i ^ 3] = ss[i]; // Reverse byte order within 4 bytes
    }

    // Append padding: 0x80 followed by zeros
    buffer[length ^ 3] = 0x80;
    for (int i = length + 1; i < l - 8; i++) {
        buffer[i ^ 3] = 0x00;
    }

    // Append original message length in bits (big-endian)
    unsigned long long bit_length = (unsigned long long)length * 8;
    for (int i = 0; i < 8; i++) {
        buffer[(l - 1 - i) ^ 3] = (bit_length >> (i * 8)) & 0xFF;
    }

    // Allocate memory for M to hold all 16-word chunks
    *M = (unsigned int*)malloc(num_blocks * 16 * sizeof(unsigned int));
    if (!*M) {
        free(buffer);
        return 0;
    }

    // Process each 64-byte chunk into 16 32-bit words
    char* buffer_ptr = buffer;
    for (int i = 0; i < num_blocks; i++, buffer_ptr += 64) {
        for (int j = 0; j < 16; j++) {
            (*M)[i * 16 + j] = ((long*)buffer_ptr)[j];
        }
    }

    free(buffer); // Free the allocated memory for the buffer

    return 1;
}
// char* random_add_str(const char* input, int num_chars) {
//     static const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);

//     int input_length = strlen(input);
//     char* result = (char*)malloc(input_length + num_chars + 1);
//     if (!result) return nullptr;

//     strcpy(result, input);
//     for (int i = 0; i < num_chars; ++i) {
//         result[input_length + i] = charset[dis(gen)];
//     }
//     result[input_length + num_chars] = '\0';

//     return result;
// }
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
    const int num_strings = 64;
    const int num_blocks = 20000;  // Adjust as per your input size
    unsigned int* h_M = (unsigned int*)malloc(num_strings * num_blocks * 16 * sizeof(unsigned int));

    char* h_sha256_outputs = (char*)malloc(num_strings * 65 * sizeof(char));
    // char* ss = "abc";
    
    char input[] = "";
    // Initialize input data (for demonstration purposes)

    for (int i = 0; i < num_strings ; i++) {
        unsigned int* temp_M = (unsigned int*)malloc( num_blocks * 16 * sizeof(unsigned int));
        char* ss = random_str(30+(num_blocks-1)*64);


        if (ss && init(ss, 30+(num_blocks-1)*64, &temp_M)) {
            // Write temp_M into h_M
            for (int j = 0; j < num_blocks * 16; j++) {
                h_M[i * num_blocks * 16 + j] = (unsigned int)temp_M[j];
            }
        }
    }
    // printf("h_m: %s\n",h_M);
    unsigned int* d_M;
    char* d_sha256_outputs;
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc((void**)&d_M, num_strings * num_blocks * 16 * sizeof(unsigned int));
    cudaMalloc((void**)&d_sha256_outputs, num_strings * 65 * sizeof(char));

    cudaMemcpy(d_M, h_M, num_strings * num_blocks * 16 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    int threads_per_block = 128;
    int num_blocks_kernel = (num_strings + threads_per_block - 1) / threads_per_block;
    cudaEventRecord(start, 0);
    int test_times=20;
    for (int j = 0; j < 1*test_times; j++){
        sha256_kernel<<<num_blocks_kernel, threads_per_block>>>(d_M, num_blocks, d_sha256_outputs, num_strings);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel  execution time: %.3f us\n", 1000*elapsedTime/test_times);
    cudaMemcpy(h_sha256_outputs, d_sha256_outputs, num_strings * 65 * sizeof(char), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < num_strings; i++) {
    //     printf("SHA-256 for string %d: %s\n", i, &h_sha256_outputs[i * 65]);
    // }

    cudaFree(d_M);
    cudaFree(d_sha256_outputs);
    free(h_M);
    free(h_sha256_outputs);

    return 0;
}
