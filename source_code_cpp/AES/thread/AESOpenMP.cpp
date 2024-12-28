//
// Created by Shengqiao Zhao on 2024-12-01.
//

#include "AESOpenMP.h"
#include "../serial/CONSTANTS.h"
#include <chrono>
#include <iostream>

#include <cstring>

unsigned char *AESOpenMP::encrypt(const unsigned char input[],
                                  const unsigned int inputLength,
                                  const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH],
                                  unsigned char* output) {
    for (unsigned int i = 0; i < inputLength; i += BLOCK_BYTES_LENGTH) {
        encryptBlock(input + i, output + i, roundKeys);
    }
    return output;
}

unsigned char *AESOpenMP::decrypt(const unsigned char input[],
                                  unsigned int inputLength,
                                  const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH],
                                  unsigned char* output) {
    for (unsigned int i = 0; i < inputLength; i += BLOCK_BYTES_LENGTH) {
        decryptBlock(input + i, output + i, roundKeys);
    }

    return output;
}

void AESOpenMP::mixColumns(unsigned char (&block)[4][4]) {
    // auto start = std::chrono::high_resolution_clock::now();

    unsigned char temp[4][4];

    for (int i = 0; i < 4; ++i) {
        memset(temp[i], 0, 4);
    }

    #pragma omp parallel for collapse(2)
    for (int rowIdx = 0; rowIdx < 4; ++rowIdx) {
        for (int colIdx = 0; colIdx < 4; ++colIdx) {
            computeGMUL(temp[colIdx][rowIdx], block, rowIdx, colIdx, MDS);
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        memcpy(block[i], temp[i], 4);
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // uint32_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // std::cout<< "mix columns takes " << duration << "ns" << std::endl;
}

void AESOpenMP::invMixColumns(unsigned char (&block)[4][4]) {
    unsigned char temp[4][4];

    for (int i = 0; i < 4; ++i) {
        memset(temp[i], 0, 4);
    }

    for (int rowIdx = 0; rowIdx < 4; ++rowIdx) {
        for (int colIdx = 0; colIdx < 4; ++colIdx) {
            computeGMUL(temp[colIdx][rowIdx], block, rowIdx, colIdx, INV_MDS);
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        memcpy(block[i], temp[i], 4);
    }
}