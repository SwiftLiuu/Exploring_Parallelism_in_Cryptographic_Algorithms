#include "cu/AEScu.h"
#include "serial/DebugUtil.h"
#include "Utils/Generator.h"
#include "Utils/sampler.h"
#include "serial/CONSTANTS.h"
#include "thread/AESOpenMP.h"

#include <omp.h>
#include <cstring>

bool resultCheck(const unsigned char* arr1, const unsigned char* arr2, const int length) {
    for (int i = 0; i < length; ++i) {
        if (arr1[i] != arr2[i]) {
            printf("idx %d, %02x != %02x\n", i, arr1[i], arr2[i]);
            return false;  // Return false if any character differs
        }
    }
    return true;  // Arrays are identical
}


int main(int argc, char *argv[])
{

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input multiplier> <thread for cpu mode> <coalesce factor> <thread per cuda block multiplier>" << std::endl;
        return 1;
    }

    int inputMultiplier = std::atoi(argv[1]);
    int threadCount = std::atoi(argv[2]);
    int coalesceFactor = std::atoi(argv[3]);
    int encryptionBlockFactor = std::atoi(argv[4]);

    omp_set_num_threads(threadCount);
    long inputLength = 16 * inputMultiplier;

    unsigned char* plain = new unsigned char[inputLength];
    Generator::generateCharArray(plain, inputMultiplier);
    std::cout << "generating " << inputLength << " byte inputs" << std::endl;

    unsigned char key[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                           0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
                           0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                           0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f};
    unsigned char roundKeys[AESSerial::TOTAL_ROUND_KEY_LENGTH];
    AESSerial::keyExpansion(key, roundKeys);

    unsigned char *d_input, *d_output, *d_round_keys;
    unsigned char *cudaEncryptionOutput = new unsigned char[inputLength];
    unsigned char *cudaDecryptoinOutput = new unsigned char[inputLength];

    // Allocate device memory
    cudaMalloc(&d_input, inputLength);
    cudaMalloc(&d_output, inputLength);
    cudaMalloc(&d_round_keys, AESSerial::TOTAL_ROUND_KEY_LENGTH * sizeof(unsigned char));
    copyToDevice((unsigned char *)S_BOX, (unsigned char *) INV_S_BOX);
    cudaMemcpy(d_round_keys, roundKeys, AESSerial::TOTAL_ROUND_KEY_LENGTH, cudaMemcpyHostToDevice);

    AESSerial aes;
    AESThreaded aesThreaded(threadCount);
    AESOpenMP AESOpenMP;
    Sampler sampler;

    unsigned char *seqEncrypted = new unsigned char[inputLength];
    unsigned char *seqDecrypted = new unsigned char[inputLength];
    unsigned char *poolEncrypted = new unsigned char[inputLength];
    unsigned char *poolDecrypted = new unsigned char[inputLength];
    unsigned char *mpEncrypted = new unsigned char[inputLength];
    unsigned char *mpDecrypted = new unsigned char[inputLength];

    const auto serialEncTime = sampler.sample(&AESSerial::encrypt, aes, plain, inputLength, roundKeys, seqEncrypted);
    std::cout << "Time consumed by the sequential encryption implementation: " << serialEncTime << "ns" << std::endl;
    const auto serialDecTime = sampler.sample(&AESSerial::decrypt, aes, seqEncrypted, inputLength, roundKeys, seqDecrypted);
    std::cout << "Time consumed by the sequential decryption implementation: " << serialDecTime << "ns" << std::endl;

    std::cout << std::endl;

    std::cout << "run threaded encrypion in " << threadCount <<" threads" << std::endl;
    const auto threadEncTime = sampler.sample(&AESThreaded::encrypt, aesThreaded, plain, inputLength, roundKeys, poolEncrypted);
    std::cout << "Time consumed by the threaded encryption implementation: " << threadEncTime << "ns" << std::endl;
    const auto threadDecTime = sampler.sample(&AESThreaded::decrypt, aesThreaded, poolEncrypted, inputLength, roundKeys, poolDecrypted);
    std::cout << "Time consumed by the threaded decryptioin implementation: " << threadDecTime << "ns" << std::endl;

    std::cout << std::endl;

    std::cout << "run openmp encrypion in " << threadCount <<" threads" << std::endl;
    const auto mpEncTime = sampler.sample(&AESOpenMP::encrypt, AESOpenMP, plain, inputLength, roundKeys, mpEncrypted);
    std::cout << "Time consumed by the openmp encryption implementation: " << mpEncTime << "ns" << std::endl;
    const auto mpDecTime = sampler.sample(&AESOpenMP::decrypt, AESOpenMP, mpEncrypted, inputLength, roundKeys, mpDecrypted);
    std::cout << "Time consumed by the openmp decryptioin implementation: " << mpDecTime << "ns" << std::endl;


    const int threadPerBlock = 16 * encryptionBlockFactor;
    const int numElementsPerBlock = threadPerBlock * coalesceFactor;
    unsigned int numBlocks = (inputLength + numElementsPerBlock - 1) / numElementsPerBlock;
    unsigned int shareMemorySize = threadPerBlock * coalesceFactor * sizeof(unsigned char);

    std::cout << std::endl;
    std::cout << "Running cuda with " << threadPerBlock << " threads per block " << std::endl;
    std::cout << "coalesce factor of " << coalesceFactor << std::endl;
    std::cout << "number of blocks " << numBlocks << std::endl;
    std::cout << "share memory size " << shareMemorySize << std::endl;

    cudaMemcpy(d_input, plain, inputLength, cudaMemcpyHostToDevice);
    const auto cudaEncTime = sampler.sample(&cudaEncryption, d_input, d_output, d_round_keys, inputLength, coalesceFactor, encryptionBlockFactor);
    std::cout << "Time consumed by the cuda encryption implementation: " << cudaEncTime << "ns" << std::endl;
    cudaMemcpy(cudaEncryptionOutput, d_output, inputLength, cudaMemcpyDeviceToHost);

    cudaMemcpy(d_input, cudaEncryptionOutput, inputLength, cudaMemcpyHostToDevice);
    const auto cudaDecTime = sampler.sample(&cudaDecryption, d_input, d_output, d_round_keys, inputLength, coalesceFactor, encryptionBlockFactor);
    std::cout << "Time consumed by the cuda decryption implementation: " << cudaDecTime << "ns" << std::endl;
    cudaMemcpy(cudaDecryptoinOutput, d_output, inputLength, cudaMemcpyDeviceToHost);

    std::cout << std::endl;

    std::cout << "\tOptimization Speedup Ratio for cuda encryption (nearest integer): "
                  << (int)((double)serialEncTime / std::max(cudaEncTime, 1u) + 0.5) << std::endl;

    std::cout << "\tOptimization Speedup Ratio for cuda decryption (nearest integer): "
              << (int)((double)serialDecTime / std::max(cudaDecTime, 1u) + 0.5) << std::endl;

    std::cout << "\tOptimization Speedup Ratio for thread encryption (nearest integer): "
                  << (int)((double)serialEncTime / std::max(threadEncTime, 1u) + 0.5) << std::endl;

    std::cout << "\tOptimization Speedup Ratio for thread decryption (nearest integer): "
              << (int)((double)serialDecTime / std::max(threadDecTime, 1u) + 0.5) << std::endl;

    std::cout << std::endl;

    if(!resultCheck(seqDecrypted, plain, inputLength)) {
        printf("sequential implementation is wrong!\n");
    }else {
        printf("sequential implementation is as expected!\n");
    }
    if(!resultCheck(seqEncrypted, poolEncrypted, inputLength)) {
        printf("pool imp does not match with serial implementation!\n");
    } else {
        printf("pool imp match with serial implementation!\n");
    }

    if(!resultCheck(seqEncrypted, cudaEncryptionOutput, inputLength)) {
        printf("cuda imp does not match with CPU implementation!\n");
    } else {
        printf("cuda imp match with CPU implementation!\n");
    }
    if(!resultCheck(seqDecrypted, cudaDecryptoinOutput, inputLength)) {
        printf("cuda imp does not decrypt correctly!\n");
    } else {
        printf("cuda imp decrypt correctly!\n");
    }

    if(!resultCheck(seqDecrypted, poolDecrypted, inputLength)) {
        printf("threaded imp does not decrypt correctly!\n");
    } else {
        printf("threaded imp decrypt correctly!\n");
    }

    if(!resultCheck(seqDecrypted, mpDecrypted, inputLength)) {
        printf("mp imp does not decrypt correctly!\n");
    } else {
        printf("mp imp decrypt correctly!\n");
    }


    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_round_keys);
    delete[] cudaEncryptionOutput;
    delete[] cudaDecryptoinOutput;
    delete[] seqEncrypted;
    delete[] seqDecrypted;
    delete[] poolEncrypted;
    delete[] poolDecrypted;
    return 0;
}
