#include "AEScu.h"

#include <cstdio>
#include <cuda_runtime.h>


#define NUMBER_OF_ROUNDS 14
#define ENCRYPTION_BLOCK_SIZE 16

// Device constants for 16x16 S_BOX, MixColumns transformation, and round keys
__constant__ unsigned char D_S_BOX[16][16];
__constant__ unsigned char D_INV_S_BOX[16][16];
__constant__ unsigned char D_MDS[4][4] = {
    {2, 3, 1, 1},
    {1, 2, 3, 1},
    {1, 1, 2, 3},
    {3, 1, 1, 2}
};

__constant__ unsigned char D_INV_MDS[4][4] = {
    {14, 11, 13, 9},
    {9, 14, 11, 13},
    {13, 9, 14, 11},
    {11, 13, 9, 14}
};

__constant__ unsigned short shiftRowLookup[16] = {0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3};
__constant__ unsigned short invShiftRowLookup[16] = {0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11};

// Helper to copy round keys and S_BOX to constant memory
void copyToDevice(const unsigned char *sBox, const unsigned char *invSBox) {
    // cudaMemcpyToSymbol(roundKeys, roundKeys, TOTAL_ROUND_KEY_LENGTH);
    cudaMemcpyToSymbol(D_S_BOX, sBox, 256);
    cudaMemcpyToSymbol(D_INV_S_BOX, invSBox, 256);
}

__device__ void printBlock (const unsigned char input[], unsigned int offset) {
    printf(" %02x %02x %02x %02x \n", input[0 + offset], input[4 + offset], input[8 + offset], input[12 + offset]);
    printf(" %02x %02x %02x %02x \n", input[1 + offset], input[5 + offset], input[9 + offset], input[13 + offset]);
    printf(" %02x %02x %02x %02x \n", input[2 + offset], input[6 + offset], input[10 + offset], input[14 + offset]);
    printf(" %02x %02x %02x %02x \n", input[3 + offset], input[7 + offset], input[11 + offset], input[15 + offset]);
}

__device__ unsigned char gmul(unsigned char a, unsigned char b) {
    unsigned char result = 0;  // Accumulator for the result
    constexpr unsigned char AES_POLY = 0x1b;  // AES irreducible polynomial (x^8 + x^4 + x^3 + x + 1)

    while (b) {  // While there are still bits in b
        if (b & 1)  // If the lowest bit of b is set
            result ^= a;  // Add (XOR) a to the result

        // Multiply a by x (left shift by 1)
        unsigned char high_bit_set = a & 0x80;  // Check if the high bit is set
        a <<= 1;  // Shift a to the left
        if (high_bit_set)  // If the high bit was set, reduce modulo AES_POLY
            a ^= AES_POLY;

        b >>= 1;  // Shift b to the right
    }

    return result;
}


// Encryption kernel
__global__ void encryptKernel(const unsigned char *input,
                              unsigned char *output,
                              const unsigned char *roundKeys,
                              const int inputSize,
                              const int coalesceFactor,
                              const int threadPerBlock) {
    extern __shared__ unsigned char block[];

    const unsigned int blockStart = threadPerBlock * coalesceFactor * blockIdx.x;
    const unsigned int blockEnd = blockStart + coalesceFactor * threadPerBlock;

    for(unsigned int inputIdx = blockStart + threadIdx.x;
            inputIdx < blockEnd;
            inputIdx += threadPerBlock) {
        unsigned char threadValue = 0x00;
        const unsigned short perBlockIdx = threadIdx.x % 16;
        const short colIdx = perBlockIdx / 4;
        const short rowIdx = perBlockIdx % 4;

        const unsigned int multiBlockOffset = (threadIdx.x / 16) * 16;

        // read and add round key
        if (inputIdx < inputSize) {
            threadValue = input[inputIdx] ^ roundKeys[perBlockIdx];
        }

        for(short round = 1; round < NUMBER_OF_ROUNDS; round++) {
            // subByte
            threadValue = D_S_BOX[ threadValue >> 4 ][ threadValue & 0xF];

            // shiftrow
            // row idx and row offset is the same for encryption
            // shift left.
            block[shiftRowLookup[perBlockIdx] + multiBlockOffset] = threadValue;
            __syncthreads();
            // mixcolumn
            threadValue = gmul(block[colIdx * 4     + multiBlockOffset], D_MDS[rowIdx][0]) ^
                          gmul(block[colIdx * 4 + 1 + multiBlockOffset], D_MDS[rowIdx][1]) ^
                          gmul(block[colIdx * 4 + 2 + multiBlockOffset], D_MDS[rowIdx][2]) ^
                          gmul(block[colIdx * 4 + 3 + multiBlockOffset], D_MDS[rowIdx][3]);

            // add round key again.
            threadValue ^= roundKeys[perBlockIdx + 16 * round];
        }

        threadValue = D_S_BOX[ threadValue >> 4 ][ threadValue & 0xF];
        block[shiftRowLookup[perBlockIdx] + multiBlockOffset] = threadValue;
        __syncthreads();
        if(inputIdx < inputSize) {
            output[inputIdx] = block[perBlockIdx + multiBlockOffset] ^ roundKeys[perBlockIdx + 16 * 14];
        }
    }

}


__global__ void decryptKernel(const unsigned char *input,
                              unsigned char *output,
                              const unsigned char *roundKeys,
                              const int inputSize,
                              const int coalesceFactor,
                              const int threadPerBlock) {
    extern __shared__ unsigned char block[];

    const unsigned int blockStart = threadPerBlock * coalesceFactor * blockIdx.x;
    const unsigned int blockEnd = blockStart + coalesceFactor * threadPerBlock;

    for(unsigned int inputIdx = blockStart + threadIdx.x;
            inputIdx < blockEnd;
            inputIdx += threadPerBlock) {
        unsigned char threadValue = 0x00;
        const unsigned short perBlockIdx = threadIdx.x % 16;
        const short colIdx = perBlockIdx / 4;
        const short rowIdx = perBlockIdx % 4;

        const unsigned int multiBlockOffset = (threadIdx.x / 16) * 16;


        // read and add round key
        if (inputIdx < inputSize) {
            threadValue = input[inputIdx] ^ roundKeys[perBlockIdx + NUMBER_OF_ROUNDS * 16];
        }

        for(short round = NUMBER_OF_ROUNDS - 1; round > 0 ; round--) {
            // inv subByte
            threadValue = D_INV_S_BOX[ threadValue >> 4 ][ threadValue & 0xF];

            // inv shiftrow + add round key
            short invIdx = invShiftRowLookup[perBlockIdx];
            threadValue ^= roundKeys[invIdx + 16 * round];
            block[invIdx + multiBlockOffset] = threadValue;
            __syncthreads();

            // mixcolumn
            threadValue = gmul(block[colIdx * 4     + multiBlockOffset], D_INV_MDS[rowIdx][0]) ^
                          gmul(block[colIdx * 4 + 1 + multiBlockOffset], D_INV_MDS[rowIdx][1]) ^
                          gmul(block[colIdx * 4 + 2 + multiBlockOffset], D_INV_MDS[rowIdx][2]) ^
                          gmul(block[colIdx * 4 + 3 + multiBlockOffset], D_INV_MDS[rowIdx][3]);
        }

        threadValue = D_INV_S_BOX[ threadValue >> 4 ][ threadValue & 0xF];
        block[invShiftRowLookup[perBlockIdx] + multiBlockOffset] = threadValue;

        __syncthreads();
        if(inputIdx < inputSize) {
            output[inputIdx] = block[perBlockIdx + multiBlockOffset] ^ roundKeys[perBlockIdx];
        }
    }
}

void cudaEncryption(const unsigned char *input,
                unsigned char *output,
                const unsigned char *roundKeys,
                const int inputSize, const int coalesceFactor, const int encryptionBlockFactor) {
    const int threadPerBlock = ENCRYPTION_BLOCK_SIZE * encryptionBlockFactor;
    const int numElementsPerBlock = threadPerBlock * coalesceFactor;
    unsigned int numBlocks = (inputSize + numElementsPerBlock - 1) / numElementsPerBlock;
    encryptKernel<<<numBlocks, threadPerBlock, threadPerBlock * coalesceFactor * sizeof(unsigned char)>>>(input, output, roundKeys, inputSize, coalesceFactor, threadPerBlock);
    cudaDeviceSynchronize();

}

void cudaDecryption(const unsigned char *input,
                unsigned char *output,
                const unsigned char *roundKeys,
                const int inputSize, const int coalesceFactor, const int encryptionBlockFactor) {
    const int threadPerBlock = ENCRYPTION_BLOCK_SIZE * encryptionBlockFactor;
    const int numElementsPerBlock = threadPerBlock * coalesceFactor;
    unsigned int numBlocks = (inputSize + numElementsPerBlock - 1) / numElementsPerBlock;
    decryptKernel<<<numBlocks, threadPerBlock, threadPerBlock * coalesceFactor * sizeof(unsigned char)>>>(input, output, roundKeys, inputSize, coalesceFactor, threadPerBlock);
    cudaDeviceSynchronize();

}
