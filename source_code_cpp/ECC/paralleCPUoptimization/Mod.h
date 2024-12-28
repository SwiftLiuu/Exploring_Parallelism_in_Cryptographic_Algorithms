class Mod {
private:
    long long mod;

public:
    Mod(long long m) {
        if (m <= 0)
            throw std::invalid_argument("ģ������Ϊ��");
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

    // ����˷������ʵ��
    inline long long mul(long long a, long long b) const {
        a = normalize(a);
        b = normalize(b);
        long long result = 0;

        // ģ��������˷�
        while (b > 0) {
            if (b & 1) { // �����ǰλΪ1���ۼ�a
                result = add(result, a);
            }
            a = add(a, a); // ģ�����ƣ�a = a * 2 % mod
            b >>= 1;       // ģ�����ƣ�b = b / 2
        }
        return result;
    }

    // ����������
    long long pow(long long a, long long n) const {
        long long res = 1;
        a = normalize(a);
        while (n) {
            if (n & 1)
                res = mul(res, a); // �۳˵�ǰ����
            a = mul(a, a);         // ����ƽ��
            n >>= 1;               // ָ������
        }
        return res;
    }

    // ��չŷ������㷨��ģ��
    long long inv(long long a) const {
        long long x, y;
        a = normalize(a);
        long long gcd = exgcd(a, mod, x, y);
        if (gcd != 1)
            throw std::runtime_error("��Ԫ������");
        return normalize(x);
    }

    // ��չŷ������㷨
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
