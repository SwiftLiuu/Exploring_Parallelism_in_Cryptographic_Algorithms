class Mod {
private:
    long long mod;

public:
    Mod(long long m) {
        if (m <= 0)
            throw std::invalid_argument("模数必须为正");
        mod = m;
    }

    inline long long normalize(long long x) const {
        x %= mod;
        if (x < 0) x += mod;
        return x;
    }

    inline long long add(long long a, long long b) const {
        return normalize(a + b);
    }

    inline long long sub(long long a, long long b) const {
        return normalize(a - b);
    }

    // 避免乘法溢出的实现
    inline long long mul(long long a, long long b) const {
        a = normalize(a);
        b = normalize(b);
        long long result = 0;

        // 模拟大整数乘法
        while (b > 0) {
            if (b & 1) { // 如果当前位为1，累加a
                result = add(result, a);
            }
            a = add(a, a); // 模拟左移，a = a * 2 % mod
            b >>= 1;       // 模拟右移，b = b / 2
        }
        return result;
    }

    // 快速幂运算
    long long pow(long long a, long long n) const {
        long long res = 1;
        a = normalize(a);
        while (n) {
            if (n & 1)
                res = mul(res, a); // 累乘当前基数
            a = mul(a, a);         // 基数平方
            n >>= 1;               // 指数右移
        }
        return res;
    }

    // 扩展欧几里得算法求模逆
    long long inv(long long a) const {
        long long x, y;
        a = normalize(a);
        long long gcd = exgcd(a, mod, x, y);
        if (gcd != 1)
            throw std::runtime_error("逆元不存在");
        return normalize(x);
    }

    // 扩展欧几里得算法
    long long exgcd(long long a, long long b, long long& x, long long& y) const {
        if (b == 0) {
            x = 1;
            y = 0;
            return a;
        }
        long long x1, y1;
        long long gcd = exgcd(b, a % b, x1, y1);
        x = y1;
        y = x1 - (a / b) * y1;
        return gcd;
    }
};
