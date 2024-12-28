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
#include <thread>
#include "Mod.h" 
#include <omp.h>
#include <random>
#include <intrin.h>
using namespace std;

int block_len = 7; 


const long long P = 1157920892373161953; 
const long long A = 3; 
const long long B = 7; 

Mod mod(P); 


struct ECPoint {
    long long x, y;
    bool is_infinity;

    ECPoint() : x(0), y(0), is_infinity(true) {}
    ECPoint(long long x, long long y) : x(x), y(y), is_infinity(false) {}
};


long long mod_sqrt_tonelli_shanks(long long a, long long p) {
    if (mod.pow(a, (p - 1) / 2) != 1) {
        throw std::runtime_error("没有模平方根");
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


ECPoint find_point_on_curve(long long x, long long &n) {
    n = 0;
    while (true) {
        long long adjusted_x = mod.add(x, n);
        long long rhs = mod.add(mod.mul(adjusted_x, mod.mul(adjusted_x, adjusted_x)),
            mod.add(mod.mul(A, adjusted_x), B));
        if (mod.pow(rhs, (P - 1) / 2) == 1) { 
            try {
                long long y = mod_sqrt_tonelli_shanks(rhs, P); 
               
                return ECPoint(adjusted_x, y);
            }
            catch (const std::runtime_error&) {
               
            }
        }
        ++n;
        if (n > P) {
            throw std::runtime_error("无法找到合法点");
        }
    }
}


ECPoint ec_point_add(const ECPoint& P, const ECPoint& Q) {
    if (P.is_infinity) return Q;
    if (Q.is_infinity) return P;

    if (P.x == Q.x && P.y == mod.sub(0, Q.y)) return ECPoint(); // 无穷远点

    long long lambda;
    if (P.x == Q.x && P.y == Q.y) {
        
        lambda = mod.mul(mod.add(mod.mul(3, mod.mul(P.x, P.x)), A), mod.inv(mod.mul(2, P.y)));
    }
    else {
        
        lambda = mod.mul(mod.sub(Q.y, P.y), mod.inv(mod.sub(Q.x, P.x)));
    }

    /*
    auto start = std::chrono::high_resolution_clock::now();*/
    long long x3 = mod.sub(mod.mul(lambda, lambda), mod.add(P.x, Q.x));
    //auto end = std::chrono::high_resolution_clock::now();
    //std::cout << "Decrypt x3 time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    long long y3 = mod.sub(mod.mul(lambda, mod.sub(P.x, x3)), P.y);
    return ECPoint(x3, y3);
}


ECPoint ec_scalar_mul(const ECPoint& P, long long k) {
    ECPoint R; 
    ECPoint Q = P;
    while (k > 0) {
        if (k % 2 == 1) {
            R = ec_point_add(R, Q);
        }
        Q = ec_point_add(Q, Q);
        k /= 2;
    }
    return R;
}

void ecc_encrypt(ECPoint& C1, ECPoint& C2, const ECPoint& plaintext, const ECPoint& G, const ECPoint& public_key, long long k) {
    #pragma omp parallel for num_threads(2)
    for (int i=0; i<2; i++) {
        if (i==0) {
            C1 = ec_scalar_mul(G, k);
        } else {
            ECPoint temp = ec_scalar_mul(public_key, k);
            C2 = ec_point_add(plaintext, temp);
        }
    }
}
void debug_point(const ECPoint& point) {
    if (point.is_infinity) {
        std::cout << "inf\n";
        return;
    }
    long long lhs = mod.mul(point.y, point.y);
    long long rhs = mod.add(mod.add(mod.mul(point.x, mod.mul(point.x, point.x)), mod.mul(A, point.x)), B);
    std::cout << "Point (" << point.x << ", " << point.y << ") - "
        << (lhs == rhs ? "in curve\n" : "not in curve\n");
}

ECPoint ecc_decrypt(const ECPoint& C1, const ECPoint& C2, long long private_key) {
    ECPoint temp = ec_scalar_mul(C1, private_key);
    temp.y = mod.sub(0, temp.y); 
    return ec_point_add(C2, temp);
}
bool is_point_on_curve(const ECPoint& point) {
    if (point.is_infinity) return true;
    long long lhs = mod.mul(point.y, point.y); 
    long long rhs = mod.add(mod.add(mod.mul(point.x, mod.mul(point.x, point.x)), mod.mul(A, point.x)), B);
    if (lhs == rhs) return true;

  
    long long neg_y = mod.sub(0, point.y);
    return mod.mul(neg_y, neg_y) == rhs;
}

size_t get_file_size(const string& file) {
	struct stat statbuf;
	stat(file.c_str(), &statbuf);
	size_t filesize = statbuf.st_size;
	return filesize;
}

long long bytes2ll(char* bytes,size_t len) {
    long long res = 0;
    for (size_t i=0; i<len; i++) {
        res = res*256 + (unsigned int)(unsigned char)bytes[i];
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
                 vector<pair<ECPoint, ECPoint>> &points,
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
    points.resize(fsize/(sizeof(long long) * 5));
    ns.resize(points.size());
    long long n;
    for (size_t i=0; i<points.size(); i++) {
        fin.read((char*)&points[i].first.x, sizeof(long long));
        fin.read((char*)&points[i].first.y, sizeof(long long));
        fin.read((char*)&points[i].second.x, sizeof(long long));
        fin.read((char*)&points[i].second.y, sizeof(long long));
        fin.read((char*)&ns[i], sizeof(long long));
        points[i].first.is_infinity = false;
        points[i].second.is_infinity = false;
    }
    fin.read((char*)&last_block_len, sizeof(int));
}
long long generate_random_k() {
    thread_local std::random_device rd; 
    thread_local std::mt19937_64 gen(rd());
    std::uniform_int_distribution<long long> dis(1, 1000000000000000000);
    return dis(gen);
}
void encrypt(const string& input_file, const string& output_file) {

    ECPoint G(1, 360265885966316755);
 
    long long private_key = 10;
    ECPoint public_key = ec_scalar_mul(G, private_key);

    ofstream fout(output_file, ios::binary);
    size_t fsize = get_file_size(input_file);
    vector<long long> data = read_data(input_file);

    vector<pair<ECPoint, ECPoint>> points(data.size());
    vector<long long> ns(data.size());
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for 
    for (int i = 0; i < data.size(); i++) {
        long long n;
        ECPoint plaintext = find_point_on_curve(data[i], n);

        long long k = generate_random_k(); 
        ecc_encrypt(points[i].first, points[i].second, plaintext, G, public_key, k);
        ns[i] = n;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Encrypt time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    for (size_t i=0; i<points.size(); i++) {
        fout.write((char*)&points[i].first.x, sizeof(long long));
        fout.write((char*)&points[i].first.y, sizeof(long long));
        fout.write((char*)&points[i].second.x, sizeof(long long));
        fout.write((char*)&points[i].second.y, sizeof(long long));
        fout.write((char*)&ns[i], sizeof(long long));
    }
    for (size_t i = 0; i < points.size(); i++) {
        const ECPoint& C1 = points[i].first;
        const ECPoint& C2 = points[i].second;

    }

    int last_block_len = fsize%block_len;
    if (last_block_len == 0 && fsize > 0) {
        last_block_len = block_len;
    }
    fout.write((char*)&last_block_len, sizeof(int));
    fout.close();
    cout << "encrypt completed, total encrypted block count: " << data.size() << ", last block length: " << last_block_len << endl;
}


void decrypt(const string& input_file, const string& output_file) {
  
    ECPoint G(1, 360265885966316755);

    long long private_key = 10;
    ECPoint public_key = ec_scalar_mul(G, private_key);

    int last_block_len;
    vector<pair<ECPoint, ECPoint>> points;
    vector<long long> ns;
    ofstream fout(output_file, ios::binary);
    read_points(input_file, points, ns, last_block_len);
    long long n;
    vector<vector<char>> bytes(points.size());
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i=0; i<points.size(); i++) {
        ECPoint C1 = points[i].first, C2 = points[i].second;
     
        ECPoint decrypted = ecc_decrypt(C1, C2, private_key);
       
        long long data = decrypted.x - ns[i];
        bytes[i].resize(block_len+3);
        ll2bytes(data, &bytes[i][0]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Decrypt time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    for (size_t i=0; i<bytes.size(); i++) {
        if (i+1 == bytes.size() && last_block_len != 0) {
            fout.write(&bytes[i][0]+block_len-last_block_len, last_block_len);
        } else {
            fout.write(&bytes[i][0], block_len);
        }
    }
    fout.close();
    cout << "decrypt completed, total decrypted block count: " << points.size()  << ", last block length: " << last_block_len << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cout << "Usage: ./ecc -t [enc|dec] -b [1~7] input_file output_file [thread_count]" << std::endl;
        return 0;
    }
    omp_set_nested(1);
    int cpu_cores = std::thread::hardware_concurrency();
    if (argc == 8) {
        omp_set_num_threads(atoi(argv[7]));
    } else if (cpu_cores > 0) {
        omp_set_num_threads(cpu_cores);
    }
	// cout << "support opm nested: " << omp_get_nested() << endl;

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
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;


    return 0;
} 