#include "AESSerial.h"

#include <cstring>
#include <chrono>
#include <iostream>


#include "../serial/CONSTANTS.h"



unsigned char *AESSerial::encrypt(const unsigned char input[],
                                  const unsigned int inputLength,
                                  const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH],
                                  unsigned char* output) {
    for (unsigned int i = 0; i < inputLength; i += BLOCK_BYTES_LENGTH) {
        encryptBlock(input + i, output + i, roundKeys);
    }
    return output;
}

unsigned char *AESSerial::decrypt(const unsigned char input[],
                                  unsigned int inputLength,
                                  const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH],
                                  unsigned char* output) {
    for (unsigned int i = 0; i < inputLength; i += BLOCK_BYTES_LENGTH) {
        decryptBlock(input + i, output + i, roundKeys);
    }

    return output;
}

void AESSerial::keyExpansion(const unsigned char key[],
                             unsigned char roundKeys[]) {

    // assign the given key to the round key first.
    for(int i = 0; i < SECRET_KEY_CHAR_LENGTH; i++) {
        roundKeys[i] = key[i];
    }
    unsigned char keyToExpand[4];

    // skip the given keys, then do the key scheduling.
    for(int i = SECRET_KEY_CHAR_LENGTH; i < TOTAL_ROUND_KEY_LENGTH; i+=4) {

        // take the previous round key
        keyToExpand[0] = roundKeys[i - 4];
        keyToExpand[1] = roundKeys[i - 3];
        keyToExpand[2] = roundKeys[i - 2];
        keyToExpand[3] = roundKeys[i - 1];

        if(i / 4 % SECRET_KEY_WORD_LENGTH == 0) {
            rotWord(keyToExpand);
            subWord(keyToExpand);

            // apply R_CON for every 8 word, which is 32 char.
            // only apply to the first char
            keyToExpand[0] ^= R_CON[i / SECRET_KEY_CHAR_LENGTH - 1];
        } else if (i / 4 % SECRET_KEY_WORD_LENGTH == 4){
            subWord(keyToExpand);
        }

        roundKeys[i]     = roundKeys[i -     SECRET_KEY_CHAR_LENGTH] ^ keyToExpand[0];
        roundKeys[i + 1] = roundKeys[i + 1 - SECRET_KEY_CHAR_LENGTH] ^ keyToExpand[1];
        roundKeys[i + 2] = roundKeys[i + 2 - SECRET_KEY_CHAR_LENGTH] ^ keyToExpand[2];
        roundKeys[i + 3] = roundKeys[i + 3 - SECRET_KEY_CHAR_LENGTH] ^ keyToExpand[3];
    }
}

void AESSerial::rotWord(unsigned char *input) {
    unsigned char temp = input[0];
    input[0] = input[1];
    input[1] = input[2];
    input[2] = input[3];
    input[3] = temp;
}

void AESSerial::subWord(unsigned char *input) {
    for (int i = 0; i < 4; i++) {
        // take top 4 bits and bottom 4 bits as the coordinate.
        input[i] = S_BOX[ input[i] >> 4 ][ input[i] & 0xF ];
    }
}

void AESSerial::encryptBlock(const unsigned char input[],
                             unsigned char output[],
                             const unsigned char *roundKeys) {
    unsigned char block[4][4];

    // take the input into 4 * 4 block
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            block[i][j] = input[i + j * 4];
        }
    }

    // DebugUtil::printValueBlock("after add round key 0", 4, block);

    addRoundKey(block, roundKeys, 0);

    for(short round  = 1; round < NUMBER_OF_ROUNDS; round++) {
        subBytes(block);
        shiftRows(block);
        mixColumns(block);
        addRoundKey(block, roundKeys, round);
    }

    subBytes(block);

    shiftRows(block);

    addRoundKey(block, roundKeys, NUMBER_OF_ROUNDS);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            output[i + 4 * j] = block[i][j];
        }
    }
}

void AESSerial::addRoundKey(unsigned char (&block)[4][4],
                            const unsigned char *key,
                            const short round) {
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            block[i][j] ^= (key + round * 16) [i + j * 4];
        }
    }
}

void AESSerial::subBytes(unsigned char (&block)[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            const unsigned char temp = block[i][j];
            block[i][j] = S_BOX[ temp >> 4 ][ temp & 0xF ];
        }
    }
}

void AESSerial::shiftRow(unsigned char (&block)[4][4],
                         const unsigned int row,
                         const unsigned int offset) {
    unsigned char temp[4];
    for(int i = 0; i < 4; i++) {
        temp[i] = block[row][ (i+offset) % 4];
    }
    memcpy(block[row], temp, 4 * sizeof(unsigned char));
}

void AESSerial::shiftRows(unsigned char (&block)[4][4]) {
    shiftRow(block, 1, 1);
    shiftRow(block, 2, 2);
    shiftRow(block, 3, 3);
}

// Galois Field Multiplication
unsigned char AESSerial::gmul(unsigned char a, unsigned char b) {
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

void AESSerial::computeGMUL(unsigned char &output,
                            const unsigned char (&block)[4][4],
                            const int rowIdx,
                            const int colIdx,
                            const unsigned char targetMDS[4][4]) {
    output = gmul(block[0][rowIdx], targetMDS[colIdx][0]) ^
             gmul(block[1][rowIdx], targetMDS[colIdx][1]) ^
             gmul(block[2][rowIdx], targetMDS[colIdx][2]) ^
             gmul(block[3][rowIdx], targetMDS[colIdx][3]);

}

void AESSerial::mixColumns(unsigned char (&block)[4][4]) {


    // auto start = std::chrono::high_resolution_clock::now();
    unsigned char temp[4][4];

    for (int i = 0; i < 4; ++i) {
        memset(temp[i], 0, 4);
    }

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



void AESSerial::decryptBlock(const unsigned char input[],
                             unsigned char output[],
                             const unsigned char *roundKeys) {
    unsigned char block[4][4];

    // take the input into 4 * 4 block
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            block[i][j] = input[i + j * 4];
        }
    }

    addRoundKey(block, roundKeys, NUMBER_OF_ROUNDS);

    // short round = NUMBER_OF_ROUNDS - 1;
    for(short round  = NUMBER_OF_ROUNDS - 1; round > 0; round--) {
        invSubBytes(block);

        invShiftRows(block);
        addRoundKey(block, roundKeys, round);

        invMixColumns(block);
    }

    invSubBytes(block);
    invShiftRows(block);
    addRoundKey(block, roundKeys, 0);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            output[i + 4 * j] = block[i][j];
        }
    }

}

void AESSerial::invSubBytes(unsigned char (&block)[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            const unsigned char temp = block[i][j];
            block[i][j] = INV_S_BOX[ temp >> 4 ][ temp & 0xF ];
        }
    }
}

void AESSerial::invShiftRows(unsigned char (&block)[4][4]) {
    shiftRow(block, 1, 3);
    shiftRow(block, 2, 2);
    shiftRow(block, 3, 1);
}

void AESSerial::invMixColumns(unsigned char (&block)[4][4]) {
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
