#ifndef DEBUGUTIL_H
#define DEBUGUTIL_H
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <vector>


class DebugUtil {
    public:

    static std::string toHex(char ch) {
        std::stringstream ss;
        ss << std::hex << std::setw(2) << std::setfill('0') << (static_cast<int>(ch) & 0xFF);
        std::string hexString = ss.str();
        return hexString;
    }

    static std::vector<std::string> fillToVec(const int length, const unsigned char input[]) {
        std::vector<char> v(length);
        std::vector<std::string> h;
        std::copy_n(input, length, v.begin());

        for (char ch : v) {
            h.push_back(toHex(ch));
        }
        return h;
    }

    static void printValue(const std::string &name, const int length, const unsigned char input[]) {
        const std::vector<std::string> v = fillToVec(length, input);

        std::cout << name << std::endl;
        for (const auto& element : v) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    static void printValueBlock(const std::string &name, const int length, unsigned char (&block)[4][4]) {
        const std::vector<std::string> v1 = fillToVec(length, block[0]);
        const std::vector<std::string> v2 = fillToVec(length, block[1]);
        const std::vector<std::string> v3 = fillToVec(length, block[2]);
        const std::vector<std::string> v4 = fillToVec(length, block[3]);

        std::cout << name << std::endl;
        for (const auto& element : v1) {
            std::cout << element << " ";
        }
        std::cout << std::endl;

        for (const auto& element : v2) {
            std::cout << element << " ";
        }
        std::cout << std::endl;

        for (const auto& element : v3) {
            std::cout << element << " ";
        }
        std::cout << std::endl;

        for (const auto& element : v4) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
};



#endif //DEBUGUTIL_H
