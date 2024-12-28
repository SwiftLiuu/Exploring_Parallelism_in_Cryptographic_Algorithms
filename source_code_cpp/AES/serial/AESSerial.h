#ifndef AESSERIAL_H
#define AESSERIAL_H


class AESSerial {
protected:
    // key expansion logic
    static void rotWord(unsigned char *input);

    static void subWord(unsigned char *input);

    // encrypt one block
    void encryptBlock(const unsigned char input[], unsigned char output[], const unsigned char *roundKeys);
    void addRoundKey(unsigned char (&block)[4][4], const unsigned char *key, short round);
    void subBytes(unsigned char (&block)[4][4]);
    void shiftRow(unsigned char (&block)[4][4], unsigned int row, unsigned int offset);
    void shiftRows(unsigned char (&block)[4][4]);

    void mixColumns(unsigned char (&block)[4][4]);

    // decrypt one block
    void decryptBlock(const unsigned char input[], unsigned char output[], const unsigned char *roundKeys);
    void invSubBytes(unsigned char (&block)[4][4]);
    void invShiftRows(unsigned char (&block)[4][4]);
    void invMixColumns(unsigned char (&block)[4][4]);

    // helper for Galois Multiplication operations
    unsigned char gmul(unsigned char a, unsigned char b);
    void computeGMUL(unsigned char &output,
                     const unsigned char (&block)[4][4],
                     int rowIdx,
                     int colIdx,
                     const unsigned char targetMDS[4][4]);


public:

    // a word is 32 bits -> 4 byte -> 4 char
    // 1 char -> 1 byte -> 8 bits
    static constexpr unsigned short NUMBER_OF_WORD = 4; // nb
    static constexpr unsigned short PER_ROUND_KEY_LENGTH = NUMBER_OF_WORD;
    static constexpr unsigned short SECRET_KEY_WORD_LENGTH = 8; // nk -> use 8 for AES 256
    static constexpr unsigned short SECRET_KEY_CHAR_LENGTH = SECRET_KEY_WORD_LENGTH * 4; // 32 char for the encryption key

    static constexpr unsigned short NUMBER_OF_ROUNDS = 14; // nr -> 256 bits key, total 14 rounds

    // the total key size of AES 256 is 240 bytes,
    // 4byte/word * 4 word * (14 round + 1 init round)
    // 16 char separation per round key.
    static constexpr unsigned short TOTAL_ROUND_KEY_LENGTH = (NUMBER_OF_ROUNDS + 1) * PER_ROUND_KEY_LENGTH * 4;
    static constexpr unsigned short BLOCK_BYTES_LENGTH = 4 * NUMBER_OF_WORD * sizeof(unsigned char);

    static void keyExpansion(const unsigned char key[], unsigned char roundKeys[]);

    // entry points
    unsigned char *encrypt(const unsigned char input[],
                           unsigned int inputLength,
                           const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH],
                           unsigned char* output);

    unsigned char *decrypt(const unsigned char input[],
                           unsigned int inputLength,
                           const unsigned char roundKeys [TOTAL_ROUND_KEY_LENGTH],
                           unsigned char* output);
};



#endif //AESSERIAL_H
