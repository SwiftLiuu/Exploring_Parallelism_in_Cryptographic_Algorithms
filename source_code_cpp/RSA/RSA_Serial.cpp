#include <iostream>
#include <fstream>  
#include <vector>   
#include <gmpxx.h>  // GMP header for C++
#include <chrono>   

using namespace std;
using namespace std::chrono;

// Modular Exponentiation Function
mpz_class modularExponentiation(mpz_class base, mpz_class exp, mpz_class mod) {
    mpz_class result = 1;
    base = base % mod; // Ensure base is within the modulus range

    while (exp > 0) {
        if (exp % 2 == 1) { // If the current bit of the exponent is set
            result = (result * base) % mod;
        }
        base = (base * base) % mod; // Square the base
        exp = exp / 2;              // Right-shift the exponent
    }
    return result;
}

int main() {
    // Define RSA Parameters as GMP integers
    mpz_class n("109120132967399429278860960508995541528237502902798129123468757937266291492576446330739696001110603907230888610072655818825358503429057592827629436413108566029093628212635953836686562675849720620786279431090218017681061521755056710823876476444260558147179707119674283982419152118103759076030616683978566631413");
    mpz_class e("65537"); // Public key exponent
    mpz_class d("46730330223584118622160180015036832148732986808519344675210555262940258739805766860224610646919605860206328024326703361630109888417839241959507572247284807035235569619173792292786907845791904955103601652822519121908367187885509270025388641700821735345222087940578381210879116823013776808975766851829020659073"); // Private key exponent

    // Read plaintext message from file
    ifstream messageFile("message.txt");
    if (!messageFile.is_open()) {
        cerr << "Error opening 'message.txt'!" << endl;
        return 1;
    }
    string message((istreambuf_iterator<char>(messageFile)), istreambuf_iterator<char>());
    messageFile.close();


    // Encrypt each character in the message
    vector<mpz_class> ciphertexts;
    auto start_encrypt = high_resolution_clock::now();
    for (char c : message) {
        mpz_class plaintext = static_cast<int>(c); // Convert char to integer
        ciphertexts.push_back(modularExponentiation(plaintext, e, n));
    }
    auto end_encrypt = high_resolution_clock::now();
    duration<double> encryption_time = end_encrypt - start_encrypt;

    // Save ciphertexts to a file
    ofstream ciphertextFile("ciphertext_serial.txt");
    if (!ciphertextFile.is_open()) {
        cerr << "Error opening 'ciphertext_serial.txt'!" << endl;
        return 1;
    }
    for (const auto &ciphertext : ciphertexts) {
        ciphertextFile << ciphertext.get_str() << " "; // Write ciphertext with space delimiter
    }
    ciphertextFile.close();
    cout << "Encrypted message saved to 'ciphertext_serial.txt'" << endl;

    // Read ciphertexts from file
    ifstream ciphertextReadFile("ciphertext_serial.txt");
    if (!ciphertextReadFile.is_open()) {
        cerr << "Error opening 'ciphertext_serial.txt'!" << endl;
        return 1;
    }
    vector<mpz_class> loadedCiphertexts;
    string ciphertextStr;
    while (ciphertextReadFile >> ciphertextStr) {
        mpz_class ciphertext;
        mpz_set_str(ciphertext.get_mpz_t(), ciphertextStr.c_str(), 10);
        loadedCiphertexts.push_back(ciphertext);
    }
    ciphertextReadFile.close();

    // Decrypt each ciphertext
    string decrypted_message;
    auto start_decrypt = high_resolution_clock::now();
    for (const auto &ciphertext : loadedCiphertexts) {
        mpz_class decrypted = modularExponentiation(ciphertext, d, n);
        decrypted_message += static_cast<char>(decrypted.get_ui()); // Convert decrypted BigInt to char
    }
    auto end_decrypt = high_resolution_clock::now();
    duration<double> decryption_time = end_decrypt - start_decrypt;

    // Save decrypted message to file
    ofstream decryptedFile("decrypted_message_serial.txt");
    if (!decryptedFile.is_open()) {
        cerr << "Error opening 'decrypted_message_serial.txt'!" << endl;
        return 1;
    }
    decryptedFile << decrypted_message;
    decryptedFile.close();
    cout << "Decrypted message saved to 'decrypted_message_serial.txt'" << endl;

    // Verify correctness
    cout << "Encryption Time: " << encryption_time.count() << " seconds" << endl;
    cout << "Decryption Time: " << decryption_time.count() << " seconds" << endl;
    cout << "Total time (encrypt + decrypt): " << (encryption_time + decryption_time).count() << " seconds" << endl;

    if (message == decrypted_message) {
        cout << "Success! Decrypted message matches the original message." << endl;
    } else {
        cout << "Error! Decrypted message does not match the original message." << endl;
    }

    return 0;
}
