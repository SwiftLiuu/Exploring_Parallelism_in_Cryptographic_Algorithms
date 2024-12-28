//
// Created by Anesthesin on 12/1/2024.
//

#ifndef GENERATOR_H
#define GENERATOR_H
#include <random>


class Generator {
    public:
    static char getRandomChar() {
        const std::string charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        static std::random_device rd;
        static std::mt19937 generator(rd());
        static std::uniform_int_distribution<> distribution(0, charset.size() - 1);

        return charset[distribution(generator)];
    }

    static void generateRandCharArray(unsigned char* array, const int n) {
        int totalSize = n * 16;
        for (int i = 0; i < totalSize; ++i) {
            array[i] = getRandomChar();  // Fill each position with a random char
        }
        array[totalSize] = '\0';  // Null-terminate the array if treating as a C-string
    }

    static void generateCharArray(unsigned char* array, const int n) {
        constexpr char pattern[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";  // Example pattern
        const int patternLength = sizeof(pattern) - 1;  // Length of pattern (exclude null terminator)

        const int totalSize = n * 16;
        for (int i = 0; i < totalSize; ++i) {
            array[i] = pattern[i % patternLength];  // Repeat pattern cyclically
        }
    }
};



#endif //GENERATOR_H
