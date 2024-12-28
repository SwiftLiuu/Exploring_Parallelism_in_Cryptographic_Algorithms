
#ifndef AESTHREADPOOL_H
#define AESTHREADPOOL_H
#include "../serial/AESSerial.h"


class AESThreaded : AESSerial {
private:
    unsigned int threadNumber;
protected:
    using AESSerial::computeGMUL;
    using AESSerial::encryptBlock;
public:
    AESThreaded(int threadNumber) : threadNumber(threadNumber) {}
    unsigned char *encrypt(const unsigned char input[],
                       unsigned int inputLength,
                       const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH], unsigned char* output);
    unsigned char *decrypt(const unsigned char input[],
                       unsigned int inputLength,
                       const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH], unsigned char* output);
};



#endif //AESTHREADPOOL_H
