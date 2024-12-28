#include <iostream>
#include <gmpxx.h> // GMP header for BigInt support
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>  // OpenMP for parallelism

using namespace std;
using namespace chrono;

// Custom Modular Exponentiation Function
mpz_class modularExponentiation(mpz_class base, mpz_class exp, mpz_class mod) {
    mpz_class result = 1;       // Initialize result to 1
    base = base % mod;          // Reduce base modulo mod

    while (exp > 0) {           // Continue until exponent becomes 0
        if (exp % 2 == 1) {     // If the current bit of exp is set
            result = (result * base) % mod; // Multiply and reduce modulo mod
        }
        base = (base * base) % mod; // Square the base and reduce modulo mod
        exp /= 2;                   
    }

    return result; // Return the final result
}

// Helper Function: Add Leading `1`
mpz_class addLeadingOne(const mpz_class &value) {
    return (value << 1) | 1; // Left shift by 1 bit, then set the LSB to 1
}

// Helper Function: Remove Leading `1`
mpz_class removeLeadingOne(const mpz_class &value) {
    return value >> 1; // Right shift by 1 bit to remove the leading `1`
}

// Helper Function: Pack Characters into BigInt
mpz_class packString(const string &message) {
    mpz_class packed = 0;
    for (char c : message) {
        packed = (packed << 7) | c; // Shift left by 7 bits and add ASCII value
    }
    return addLeadingOne(packed); // Add leading `1` for safety
}

// Helper Function: Unpack BigInt into Characters
string unpackBigInt(const mpz_class &packed) {
    mpz_class unpacked = removeLeadingOne(packed); // Remove leading `1`
    string message;
    while (unpacked > 0) {
        mpz_class extracted = (unpacked & 127);  // Extract the last 7 bits
        char c = extracted.get_ui();            // Convert to a char
        message = c + message;                  // Prepend the character to the message
        unpacked >>= 7;                         // Shift right by 7 bits
    }
    return message;
}

// Helper Function: Calculate Maximum Number of Characters
size_t calculateMaxChars(const mpz_class &n) {
    size_t bit_length = mpz_sizeinbase(n.get_mpz_t(), 2); // Bit length of n
    return (bit_length - 1) / 7; // Floor((bit_length - 1) / 7)
}

int main() {
    // Set the number of threads
    int num_threads = 22;
    omp_set_num_threads(num_threads);

    // Define RSA Parameters as GMP integers
    mpz_class n("109120132967399429278860960508995541528237502902798129123468757937266291492576446330739696001110603907230888610072655818825358503429057592827629436413108566029093628212635953836686562675849720620786279431090218017681061521755056710823876476444260558147179707119674283982419152118103759076030616683978566631413");
    mpz_class e("65537"); // encryption key
    mpz_class d("46730330223584118622160180015036832148732986808519344675210555262940258739805766860224610646919605860206328024326703361630109888417839241959507572247284807035235569619173792292786907845791904955103601652822519121908367187885509270025388641700821735345222087940578381210879116823013776808975766851829020659073"); // Private key 

    
    ifstream messageFile("message.txt");
    string message((istreambuf_iterator<char>(messageFile)), istreambuf_iterator<char>());
    messageFile.close();


    // Calculate the maximum number of characters per chunk
    size_t max_chars = calculateMaxChars(n);

    // Split the message into chunks
    vector<string> chunks;
    for (size_t i = 0; i < message.size(); i += max_chars) {
        chunks.push_back(message.substr(i, max_chars));
    }

    // Encrypt each chunk in parallel
    vector<mpz_class> ciphertexts(chunks.size());
    auto start_encrypt = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < chunks.size(); ++i) {
        mpz_class packed = packString(chunks[i]); // Pack chunk into BigInt
        ciphertexts[i] = modularExponentiation(packed, e, n); // Encrypt using custom function
    }
    auto end_encrypt = high_resolution_clock::now();
    duration<double> encryption_time = end_encrypt - start_encrypt;

    // Save ciphertexts to a file
    ofstream ciphertextFile("ciphertext_parallel.txt");
    for (const auto &ciphertext : ciphertexts) {
        ciphertextFile << ciphertext.get_str() << " "; // Write ciphertext with a space delimiter
    }
    ciphertextFile.close();
    cout << "Message encrypted and saved to 'ciphertext_parallel.txt'" << endl;

    // Read the ciphertext from the file
    ifstream readCiphertextFile("ciphertext_parallel.txt");
    vector<mpz_class> loadedCiphertexts;
    string ciphertextStr;

    while (readCiphertextFile >> ciphertextStr) {
        mpz_class ciphertext;
        mpz_set_str(ciphertext.get_mpz_t(), ciphertextStr.c_str(), 10);
        loadedCiphertexts.push_back(ciphertext);
    }
    readCiphertextFile.close();

    // Decrypt each chunk in parallel
    vector<string> decrypted_chunks(loadedCiphertexts.size());
    auto start_decrypt = high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < loadedCiphertexts.size(); ++i) {
        mpz_class decrypted_packed = modularExponentiation(loadedCiphertexts[i], d, n); // Decrypt using custom function
        decrypted_chunks[i] = unpackBigInt(decrypted_packed); // Unpack BigInt to string
    }
    auto end_decrypt = high_resolution_clock::now();
    duration<double> decryption_time = end_decrypt - start_decrypt;

    // Reconstruct the decrypted message
    string decrypted_message;
    for (const string &chunk : decrypted_chunks) {
        decrypted_message += chunk;
    }

    // Save the decrypted message to a file
    ofstream decryptedFile("decrypted_message_parallel.txt");
    decryptedFile << decrypted_message;
    decryptedFile.close();

    cout << "Decrypted message saved to 'decrypted_message_parallel.txt'" << endl;

    // Print the required information
    cout << "Number of threads: " << num_threads << endl;
    cout << "Time consumed for encryption: " << encryption_time.count() << " seconds" << endl;
    cout << "Time consumed for decryption: " << decryption_time.count() << " seconds" << endl;
    cout << "Total time (encrypt + decrypt): " << (encryption_time + decryption_time).count() << " seconds" << endl;

    // Verify correctness
    if (message == decrypted_message) {
        cout << "Success! Decrypted message matches the original message." << endl;
    } else {
        cout << "Error! Decrypted message does not match the original message." << endl;
    }

    return 0;
}
