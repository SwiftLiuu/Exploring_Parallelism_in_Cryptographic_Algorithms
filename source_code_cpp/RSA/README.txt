# RSA Implementation

This repository contains two implementations of the RSA encryption algorithm in C++:

1. **RSA_Serial**: This version implements RSA without packing optimization. It processes the message character by character in a serial manner.
2. **RSA_Packing**: This version includes packing optimization and parallel processing. It groups multiple characters into a single big integer for more efficient encryption and decryption. And the number of threads can be tuned inside the code.

Both implementations read the plaintext message from `message.txt` and, upon execution, generate the following files:
- `ciphertext_serial.txt` or `ciphertext_parallel.txt` for the encrypted message.
- `decrypted_message_serial.txt` or `decrypted_message_parallel.txt` for the decrypted message.

## Prerequisites

To compile and run these implementations, the following dependencies must be installed:

1. **OpenMP**: Required for parallel processing in the RSA_Packing implementation.
2. **Big Integer Library**: The `mpz_class` type from the GMP (GNU Multiple Precision Arithmetic Library) is used to handle large integers.

### Installation Steps

1. Install OpenMP:
   ```bash
   sudo apt-get install libomp-dev
