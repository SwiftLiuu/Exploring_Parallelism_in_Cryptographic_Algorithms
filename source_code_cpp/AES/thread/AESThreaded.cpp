
#include "AESThreaded.h"

#include <thread>
#include <vector>

#include "../serial/CONSTANTS.h"

unsigned char * AESThreaded::encrypt(const unsigned char input[],
                                       const unsigned int inputLength,
                                       const unsigned char roundKeys[TOTAL_ROUND_KEY_LENGTH],
                                       unsigned char* output) {
    std::vector<std::thread> threads;
    unsigned int blocksPerThread = inputLength / (BLOCK_BYTES_LENGTH * AESThreaded::threadNumber);

    for (unsigned int t = 0; t < AESThreaded::threadNumber; ++t) {
        unsigned int startBlock = t * blocksPerThread * BLOCK_BYTES_LENGTH;
        unsigned int endBlock = (t == AESThreaded::threadNumber - 1)
            ? inputLength
            : startBlock + blocksPerThread * BLOCK_BYTES_LENGTH;

        threads.emplace_back([this, &input, &output, &roundKeys, startBlock, endBlock]() {
            for (unsigned int i = startBlock; i < endBlock; i += BLOCK_BYTES_LENGTH) {
                encryptBlock(input + i, output + i, roundKeys);
            }
        });
    }

    // Join all threads
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    return output;

}

unsigned char *AESThreaded::decrypt(const unsigned char input[],
                       unsigned int inputLength,
                       const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH],
                       unsigned char* output) {
    std::vector<std::thread> threads;
    unsigned int blocksPerThread = inputLength / (BLOCK_BYTES_LENGTH * AESThreaded::threadNumber);

    for (unsigned int t = 0; t < AESThreaded::threadNumber; ++t) {
        unsigned int startBlock = t * blocksPerThread * BLOCK_BYTES_LENGTH;
        unsigned int endBlock = (t == AESThreaded::threadNumber - 1)
            ? inputLength
            : startBlock + blocksPerThread * BLOCK_BYTES_LENGTH;

        threads.emplace_back([this, &input, &output, &roundKeys, startBlock, endBlock]() {
            for (unsigned int i = startBlock; i < endBlock; i += BLOCK_BYTES_LENGTH) {
                decryptBlock(input + i, output + i, roundKeys);
            }
        });
    }


    // Join all threads
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    return output;
}
