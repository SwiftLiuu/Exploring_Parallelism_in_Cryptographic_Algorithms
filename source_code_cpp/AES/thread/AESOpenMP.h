//
// Created by Shengqiao Zhao on 2024-12-01.
//

#ifndef AESOPENMP_H
#define AESOPENMP_H
#include "../serial/AESSerial.h"


class AESOpenMP : AESSerial {
protected:
    using AESSerial::computeGMUL;
    using AESSerial::encryptBlock;

    void mixColumns(unsigned char (&block)[4][4]);
    void invMixColumns(unsigned char (&block)[4][4]);

public:
    unsigned char *encrypt(const unsigned char input[],
                       unsigned int inputLength,
                       const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH], unsigned char* output);
    unsigned char *decrypt(const unsigned char input[],
                       unsigned int inputLength,
                       const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH], unsigned char* output);
};



#endif //AESOPENMP_H
