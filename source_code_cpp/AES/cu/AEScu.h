#ifndef AESCU_H
#define AESCU_H

void copyToDevice(const unsigned char *sBox, const unsigned char *invSBox);
void cudaEncryption(const unsigned char *input, unsigned char *output, const unsigned char *roundKeys, int inputSize, int coalesceFactor, int encryptionBlockFactor);
void cudaDecryption(const unsigned char *input, unsigned char *output, const unsigned char *roundKeys, int inputSize, int coalesceFactor, int encryptionBlockFactor);

#endif
