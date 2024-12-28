#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <cmath>
#include <sys/stat.h>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d, code=%d (%s)\n", __FILE__, __LINE__, static_cast<int>(err), cudaGetErrorString(err)); \
            std::cout << "Exiting at line " << __LINE__ << " due to error." << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            printf("Kernel launch error at %s:%d, code=%d (%s)\n", __FILE__, __LINE__, static_cast<int>(err), cudaGetErrorString(err)); \
            std::cout << "Kernel launch failed at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            printf("Kernel execution error at %s:%d, code=%d (%s)\n", __FILE__, __LINE__, static_cast<int>(err), cudaGetErrorString(err)); \
            std::cout << "Kernel execution failed at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


class Mod {
private:
    long long mod;

public:
    __device__ Mod(long long m) {

        mod = m;
    }

    __device__ inline long long normalize(long long x) const {
        x %= mod;
        if (x < 0) x += mod;
        return x;
    }

    __device__ inline long long add(long long a, long long b) const {
        return normalize(a + b);
    }

    __device__ inline long long sub(long long a, long long b) const {
        return normalize(a - b);
    }

 
    __device__ inline long long mul(long long a, long long b) const {
        a = normalize(a);
        b = normalize(b);
        long long result = 0;

        
        while (b > 0) {
            if (b & 1) {
                result = add(result, a);
            }
            a = add(a, a); 
            b >>= 1;      
        }
        return result;
    }

             
    __device__ long long pow(long long a, long long n) const {
        long long res = 1;
        a = normalize(a);
        while (n) {
            if (n & 1)
                res = mul(res, a);
            a = mul(a, a);         
            n >>= 1;              
        }
        return res;
    }


    __device__ long long inv(long long a)  {
        long long x, y;
        a = normalize(a);
        long long gcd = exgcd(a, mod, x, y);

 
        return normalize(x);
    }


    __device__ long long exgcd(long long a, long long b, long long& x, long long& y) {
        long long x0 = 1, y0 = 0;
        long long x1 = 0, y1 = 1;
        long long q, temp;
        while (b != 0) {
            q = a / b;
            temp = a % b;
            a = b;
            b = temp;

            temp = x1;
            x1 = x0 - q * x1;
            x0 = temp;

            temp = y1;
            y1 = y0 - q * y1;
            y0 = temp;
        }
        x = x0;
        y = y0;
        return a;
    }


};


int block_len = 7; //  


#define P_VALUE 1157920892373161953
#define A_VALUE 3  
#define B_VALUE 7

struct ECPoint {
    long long x, y;
    bool is_infinity;

    __host__ __device__ ECPoint() : x(0), y(0), is_infinity(true) {}
    __host__ __device__ ECPoint(long long x, long long y) : x(x), y(y), is_infinity(false) {}
};


__device__ long long mod_sqrt_tonelli_shanks(long long a, long long p, Mod mod) {
    if (mod.pow(a, (p - 1) / 2) != 1) {

    }

    if (p % 4 == 3) { 
        return mod.pow(a, (p + 1) / 4);
    }

    
    long long Q = p - 1;
    long long S = 0;
    while (Q % 2 == 0) {
        Q /= 2;
        S++;
    }

   
    long long z = 2;
    while (mod.pow(z, (p - 1) / 2) == 1) {
        z++;
    }

    long long M = S;
    long long c = mod.pow(z, Q);
    long long t = mod.pow(a, Q);
    long long R = mod.pow(a, (Q + 1) / 2);

    while (t != 0 && t != 1) {
        long long i = 0, temp = t;
        while (temp != 1 && i < M) {
            temp = mod.mul(temp, temp);
            i++;
        }
        long long b = mod.pow(c, 1 << (M - i - 1));
        M = i;
        c = mod.mul(b, b);
        t = mod.mul(t, c);
        R = mod.mul(R, b);
    }

    return R;
}

__device__ ECPoint find_point_on_curve(long long x, long long &n, Mod mod) {
    n = 0;
    int max_iterations = 1000;
    while (n <= max_iterations) {
        long long adjusted_x = mod.add(x, n);
        long long rhs = mod.add(
            mod.mul(adjusted_x, mod.mul(adjusted_x, adjusted_x)),
            mod.add(mod.mul(A_VALUE, adjusted_x), B_VALUE));

        
        if (mod.pow(rhs, (P_VALUE - 1) / 2) == 1) {
            long long y = mod_sqrt_tonelli_shanks(rhs, P_VALUE, mod); 
            

            return ECPoint(adjusted_x, y);
        }

        

        ++n;
    }

   
    printf("Unable to find a valid point within %d iterations, exiting...\n", max_iterations);
    return ECPoint(0, 0); 
}




__device__ struct ECPoint ec_point_add(struct ECPoint P, struct ECPoint Q, Mod mod) {
    if (P.is_infinity) return Q;  
    if (Q.is_infinity) return P;  

    if (P.x == Q.x && P.y == mod.sub(0, Q.y)) {
       
        return ECPoint();
    }

    long long lambda;  
    if (P.x == Q.x && P.y == Q.y) {
       
        long long P_x_squared = mod.mul(P.x, P.x);  
        long long numerator = mod.add(mod.mul(3, P_x_squared), A_VALUE);  
        long long denominator = mod.mul(2, P.y);  
        if (denominator == 0) {
            printf("Error: Division by zero in point doubling\n");
            return ECPoint(); 
        }
        long long denominator_inv = mod.inv(denominator); 
        lambda = mod.mul(numerator, denominator_inv);  

      

    } else {
     
        long long numerator = mod.sub(Q.y, P.y); 
        long long denominator = mod.sub(Q.x, P.x);  
        if (denominator == 0) {
            printf("Error: Division by zero in point addition\n");
            return ECPoint();
        }
        long long denominator_inv = mod.inv(denominator);
        lambda = mod.mul(numerator, denominator_inv);  

     

    }

    long long lambda_squared = mod.mul(lambda, lambda);  
    long long x3 = mod.sub(lambda_squared, mod.add(P.x, Q.x)); 
    long long temp = mod.sub(P.x, x3);  
    long long y3 = mod.sub(mod.mul(lambda, temp), P.y); 

  


    return ECPoint(x3, y3);
}


__device__ struct ECPoint ec_scalar_mul(struct ECPoint P, long long k, Mod mod) {
    ECPoint R; 
    ECPoint Q = P;
    while (k > 0) {
        if (k % 2 == 1) {
            R = ec_point_add(R, Q, mod);
        }
        Q = ec_point_add(Q, Q, mod);
        k /= 2;
    }
    return R;
}

  
__device__ void ecc_encrypt(struct ECPoint* C1, struct ECPoint* C2,
                            struct ECPoint plaintext,  struct ECPoint G,
                            struct ECPoint public_key, long long k, Mod mod) {
    *C1 = ec_scalar_mul(G, k, mod);
    ECPoint temp = ec_scalar_mul(public_key, k, mod);
    *C2 = ec_point_add(plaintext, temp, mod);
}


__device__ struct ECPoint ecc_decrypt(struct ECPoint C1, struct ECPoint C2, long long private_key, Mod mod) {
    ECPoint temp = ec_scalar_mul(C1, private_key, mod);
    temp.y = mod.sub(0, temp.y); //   
    return ec_point_add(C2, temp, mod);
}

size_t get_file_size(const string& file) {
	struct stat statbuf;
	stat(file.c_str(), &statbuf);
	size_t filesize = statbuf.st_size;
	return filesize;
}

long long bytes2ll(char* bytes, size_t len) {
    long long res = 0;

    for (size_t i = 0; i < len; i++) {
       
        if (res > (LLONG_MAX / 256)) {
            printf("Error: Value out of range during calculation. Current res: %lld\n", res);
            exit(EXIT_FAILURE);
        }
        res = res * 256 + (unsigned int)(unsigned char)bytes[i];

     
        if (res > LLONG_MAX || res < LLONG_MIN) {
            printf("Error: Value out of range after addition. Current res: %lld\n", res);
            exit(EXIT_FAILURE);
        }
    }
    return res;
}


void ll2bytes(long long n, char* bytes) {
    memset(bytes, 0, block_len);
    int idx = block_len - 1;
    while (n != 0) {
        bytes[idx--] = (unsigned char)(n&255);
        n = (((unsigned long long)n)>>8);
    }
}

vector<long long> read_data(const string& input_file) {
    ifstream fin(input_file, ios::binary);
    if (!fin) {
        cout << "can't open file " << input_file << endl;
        exit(1);
    }
    vector<long long> res;
    char buf[50];
    while (true) {
        fin.read(buf, block_len);
        size_t cnt = fin.gcount();
        if (cnt == 0) {
            break;
        }
        buf[cnt] = 0;
        res.push_back(bytes2ll(buf,cnt));
    }
    return res;
}

void read_points(const string& input_file,
                 vector<ECPoint> &points,
                 vector<long long> &ns,
                 int &last_block_len) {
    ifstream fin(input_file, ios::binary);
    size_t fsize;
    if (!fin) {
        cout << "can't open file " << input_file << endl;
        exit(1);
    }
    fin.seekg(0, ios::end);
    fsize = fin.tellg();
    fin.seekg(0, ios::beg);
    ns.resize(fsize/(sizeof(long long) * 5));
    points.resize(ns.size()*2);
    
    long long n;
    for (size_t i=0; i<ns.size(); i++) {
        fin.read((char*)&points[i*2].x, sizeof(long long));
        fin.read((char*)&points[i*2].y, sizeof(long long));
        fin.read((char*)&points[i*2+1].x, sizeof(long long));
        fin.read((char*)&points[i*2+1].y, sizeof(long long));
        fin.read((char*)&ns[i], sizeof(long long));
        points[i*2].is_infinity = false;
        points[i*2+1].is_infinity = false;
    }
    fin.read((char*)&last_block_len, sizeof(int));
}

__global__ void ecc_encrypt_kernel(
        long long *d_data, int size, long long k, ECPoint* points, long long* ns)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        Mod mod(P_VALUE);
        ECPoint G(1, 360265885966316755);
        long long private_key = 10;
        ECPoint public_key = ec_scalar_mul(G, private_key, mod);

        long long local_n = 0;
        ECPoint plaintext = find_point_on_curve(d_data[tid], local_n, mod);
        ns[tid] = local_n;
      
        ecc_encrypt(&points[tid * 2], &points[tid * 2 + 1], plaintext, G, public_key, k, mod);

   
    }
}


void encrypt(const string& input_file, const string& output_file) {
    ofstream fout(output_file, ios::binary);
    size_t fsize = get_file_size(input_file);
    vector<long long> data = read_data(input_file);

    vector<ECPoint> points(data.size()*2);
    vector<long long> ns(data.size());
    long long *d_data, *d_ns;
    ECPoint *d_points;
    size_t bytes = data.size()* sizeof(long long);
    int threads_per_block = 256;
    int blocks = (data.size() + threads_per_block - 1)/threads_per_block;
    long long k = 10; //      
    auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_ns, bytes));
    CUDA_CHECK(cudaMalloc(&d_points, 2*data.size()*sizeof(ECPoint)));
    cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice);
    ecc_encrypt_kernel<<<blocks, threads_per_block>>>(d_data, data.size(), k,  d_points, d_ns);
    CUDA_KERNEL_CHECK();
    cudaMemcpy(points.data(), d_points, 2*data.size()*sizeof(ECPoint), cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.data(), d_ns, data.size()*sizeof(long long), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Encrypt time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    for (size_t i=0; i<data.size(); i++) {
        fout.write((char*)&points[i*2].x, sizeof(long long));
        fout.write((char*)&points[i*2].y, sizeof(long long));
        fout.write((char*)&points[i*2+1].x, sizeof(long long));
        fout.write((char*)&points[i*2+1].y, sizeof(long long));
        fout.write((char*)&ns[i], sizeof(long long));
    }

    int last_block_len = fsize%block_len;
    if (last_block_len == 0 && fsize > 0) {
        last_block_len = block_len;
    }
    fout.write((char*)&last_block_len, sizeof(int));
    fout.close();
    cout << "encrypt completed, total encrypted block count: " << data.size() << ", last block length: " << last_block_len << endl;
}

__global__ void ecc_decrypt_kernel(
    ECPoint* points, long long* ns, int size,
    long long private_key, long long *d_data)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < size) {
        Mod mod(P_VALUE);
        ECPoint decrypted = ecc_decrypt(points[tid*2], points[tid*2+1], private_key, mod);
 
        d_data[tid] = decrypted.x - ns[tid];
    }
}

void decrypt(const string& input_file, const string& output_file) {
 
    ECPoint G(1, 360265885966316755);
 
    long long private_key = 10;

    int last_block_len;
    vector<ECPoint> points;
    vector<long long> ns;
    ofstream fout(output_file, ios::binary);
    read_points(input_file, points, ns, last_block_len);
    cout << "??????: " << ", last block length: " << last_block_len << endl;
    int threads_per_block = 256;
    int blocks = (ns.size() + threads_per_block - 1)/threads_per_block;
    long long *d_data;
    long long *d_ns;
    ECPoint *d_points;
    vector<long long> data(ns.size());
    CUDA_CHECK(cudaMalloc(&d_data, points.size()/2*sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_ns, ns.size()*sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_points, points.size()*sizeof(ECPoint)));
    cudaMemcpy(d_points, points.data(), points.size()*sizeof(ECPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ns, ns.data(), ns.size()*sizeof(long long), cudaMemcpyHostToDevice);
 
    auto start = std::chrono::high_resolution_clock::now();
    ecc_decrypt_kernel<<<blocks, threads_per_block>>>(d_points, d_ns, ns.size(), private_key, d_data);
    CUDA_KERNEL_CHECK();
    cudaMemcpy(data.data(), d_data, data.size()*sizeof(long long), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Decrypt time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    for (size_t i=0; i<data.size(); i++) {
        char bytes[50];
        ll2bytes(data[i], bytes);
        if (i+1 == data.size() && last_block_len != 0) {
            fout.write(bytes+block_len-last_block_len, last_block_len);
        } else {
            fout.write(bytes, block_len);
        }
    }
    fout.close();
    cout << "decrypt completed, total decrypted block count: " << points.size()  << ", last block length: " << last_block_len << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cout << "Usage: ./ecc -t [enc|dec] -b [1~7] input_file output_file" << std::endl;
        return 0;
    }
    auto start = std::chrono::high_resolution_clock::now();
    block_len = atoi(argv[4]);
    if (!strcmp(argv[2], "enc")) {
        encrypt(argv[5], argv[6]);
    } else if (!strcmp(argv[2], "dec")) {
        decrypt(argv[5], argv[6]);
    } else {
        std::cout << "error value of -t, which must be enc or dec" << std::endl;
        return 0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    
    return 0;
}  