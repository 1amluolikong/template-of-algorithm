#include<iostream>
#include<vector>
#include<cmath>
#include<ranges>
#include<bit>
#include<algorithm>
#include<cstdio>
#include<unordered_map>
#include<map>
#include<string>
#include<tuple>
#include<queue>
#include<numeric>
#include<unordered_set>
#include<stack>
#include<bitset>
#include<climits>

// quick flip binary bit
uint bitreverse32(uint n) {
    const uint32_t M1 = 0x55555555; // 01010101010101010101010101010101
    const uint32_t M2 = 0x33333333; // 00110011001100110011001100110011
    const uint32_t M4 = 0x0f0f0f0f; // 00001111000011110000111100001111
    const uint32_t M8 = 0x00ff00ff; // 00000000111111110000000011111111

    n = n >> 1 & M1 | (n & M1) << 1; // >> > & > |
    n = n >> 2 & M2 | (n & M2) << 2;
    n = n >> 4 & M4 | (n & M4) << 4;
    n = n >> 8 & M8 | (n & M8) << 8;
    return n >> 16 | n << 16;
}

// 回滚莫队
void solve() {
    int n; cin >> n;
    int block = ceil(n / sqrt(n * 2));
    int max_cnt = 0, min_val = 0;
    unordered_map<int, int> cnt;
    auto add = [&](int x) {
        int c = ++cnt[x];
        if (c > max_cnt) {
            max_cnt = c;
            min_val = x;
        }
        else if (c == max_cnt) {
            min_val = min(min_val, x);
        }
        };
    vector<int> a(n);
    for (int& x : a) cin >> x;
    struct query {
        int bid, l, r, qid;
    };
    vector<query> qs;
    vector<int> ans(n);
    for (int i = 0; i < n; i++) {
        int l, r; cin >> l >> r;
        l--;
        if (r - l > block) {
            qs.emplace_back(l / block, l, r, i);
            continue;
        }
        for (int j = l; j < r; j++) add(a[j]);
        ans[i] = min_val;
        cnt.clear();
        max_cnt = 0;
    }
    ranges::sort(qs, {}, [&](const auto& q) { return pair(q.bid, q.r); });
    int r = 0;
    for (int i = 0; i < qs.size(); i++) {
        auto& q = qs[i];
        int b = (q.bid + 1) * block;
        if (i == 0 || q.bid > qs[i - 1].bid) {
            r = b;
            cnt.clear();
            max_cnt = 0;
        }
        for (; r < q.r; r++) {
            add(a[r]);
        }
        int tmp_max_cnt = max_cnt, tmp_min_val = min_val;
        for (int j = q.l; j < b; j++) add(a[j]);
        ans[q.qid] = min_val;
        max_cnt = tmp_max_cnt, min_val = tmp_min_val;
        for (int j = q.l; j < b; j++) cnt[a[j]]--;

    }
    for (int& x : ans) cout << x << '\n';
}

// 用贡献法计算符合要求的数字的和
long long digitDPContribution(long long low, long long high, int k) {
    string low_s = to_string(low);
    string high_s = to_string(high);
    int n = high_s.size();
    int diff_lh = n - low_s.size();
    vector memo(n, vector<pair<long long, long long>>(1 << 10, { -1, -1 }));

    // dfs 返回两个数：子树合法数字个数，子树数位总和
    auto dfs = [&](this auto&& dfs, int i, int mask, bool limit_low, bool limit_high) -> pair<long long, long long> {
        if (i == n) {
            // 如果没有特殊约束，那么能递归到终点的都是合法数字
            return { 1, 0 };
        }

        if (!limit_low && !limit_high && memo[i][mask].first >= 0) {
            return memo[i][mask];
        }

        int lo = limit_low && i >= diff_lh ? low_s[i - diff_lh] - '0' : 0;
        int hi = limit_high ? high_s[i] - '0' : 9;

        long long cnt = 0, sum = 0;
        int d = lo;

        // 如果前导零不影响答案，去掉这个 if block
        if (limit_low && i < diff_lh) {
            // 不填数字，上界不受约束
            tie(cnt, sum) = dfs(i + 1, 0, true, false);
            d = 1;
        }

        for (; d <= hi; d++) {
            int new_mask = mask | 1 << d;
            if (popcount((uint32_t)new_mask) > k) { // 不满足要求
                continue;
            }
            auto [sub_cnt, sub_sum] = dfs(i + 1, new_mask, limit_low && d == lo, limit_high && d == hi);
            cnt += sub_cnt; // 累加子树的合法数字个数
            sum += sub_sum; // 累加子树的数位总和
            sum += d * sub_cnt; // d 会出现在 sub_cnt 个数中（贡献法）
            // cnt %= MOD; sum %= MOD;
        }

        pair<long long, long long> res = { cnt, sum };
        if (!limit_low && !limit_high) {
            memo[i][mask] = res;
        }
        return res;
        };

    return dfs(0, 0, true, true).second;
}

// to get number of valid scheme
long long digitDP(long long low, long long high, int target) {
    string low_s = to_string(low);
    string high_s = to_string(high);
    int n = high_s.size();
    int diff_lh = n - low_s.size();
    vector memo(n, vector<long long>(target + 1, -1));

    auto dfs = [&](this auto&& dfs, int i, int cnt0, bool limit_low, bool limit_high) -> long long {
        if (cnt0 > target) {
            return 0; // 不合法
        }
        if (i == n) {
            return cnt0 == target;
        }

        if (!limit_low && !limit_high && memo[i][cnt0] >= 0) {
            return memo[i][cnt0];
        }

        int lo = limit_low && i >= diff_lh ? low_s[i - diff_lh] - '0' : 0;
        int hi = limit_high ? high_s[i] - '0' : 9;

        long long res = 0;
        int d = lo;

        // 通过 limit_low 和 i 可以判断能否不填数字，无需 is_num 参数
        // 如果前导零不影响答案，去掉这个 if block
        if (limit_low && i < diff_lh) {
            // 不填数字，上界不受约束
            res = dfs(i + 1, 0, true, false);
            d = 1;
        }

        for (; d <= hi; d++) {
            // 统计 0 的个数
            res += dfs(i + 1, cnt0 + (d == 0), limit_low && d == lo, limit_high && d == hi);
            // res %= MOD;
        }

        if (!limit_low && !limit_high) {
            memo[i][cnt0] = res;
        }
        return res;
        };

    return dfs(0, 0, true, true);
}

struct TupleHash {
    template<typename T>
    static void hash_combine(size_t& seed, const T& v) {
        // 参考 boost::hash_combine
        seed ^= hash<T>{}(v)+0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    template<typename Tuple, size_t Index = 0>
    static void hash_tuple(size_t& seed, const Tuple& t) {
        if constexpr (Index < tuple_size_v<Tuple>) {
            hash_combine(seed, get<Index>(t));
            hash_tuple<Tuple, Index + 1>(seed, t);
        }
    }

    template<typename... Ts>
    size_t operator()(const tuple<Ts...>& t) const {
        size_t seed = 0;
        hash_tuple(seed, t);
        return seed;
    }
};

class seg {
    vector<int> t;
public:
    seg(vector<int>& a) {
        unsigned n = a.size();
        t.resize(2 << bit_width(n - 1));
        build(a, 1, 0, n - 1);
    }

    void maintain(int o) {
        t[o] = max(t[2 * o], t[2 * o + 1]);
    }

    void build(const vector<int>& a, int o, int l, int r) {
        if (l == r) {
            t[o] = a[l];
            return;
        }
        int mid = (l + r) / 2;
        build(a, 2 * o, l, mid);
        build(a, 2 * o + 1, mid + 1, r);
        maintain(o);
    }

    int find(int o, int l, int r, int x, int idx) {
        if (t[o] <= x || idx > r) return -1;
        if (l == r) {
            return l;
        }
        int mid = (l + r) / 2;
        int i = find(2 * o, l, mid, x, idx);
        if (i < 0) {
            i = find(2 * o + 1, mid + 1, r, x, idx);
        }
        return i;
    }
};

class lazy_seg {
    using T = pair<int, int>;
    using F = int;

    const F TODO_INIT = 0;

    struct node {
        T val;
        F todo;
    };

    int n;
    vector<node> t;

    int merge_todo(const F& a, const F& b) const {
        return a + b;
    }

    T merge_val(const T& a, const T& b) const {
        return { min(a.first, b.first), max(a.second, b.second) };
    }

    void maintain(int o) {
        t[o].val = merge_val(t[2 * o].val, t[2 * o + 1].val);
    }

    void apply(int o, int l, int r, F todo) {
        node& cur = t[o];
        cur.val.first += todo;
        cur.val.second += todo;
        cur.todo = merge_todo(todo, cur.todo);
    }

    void spread(int o, int l, int r) {
        node& cur = t[o];
        F todo = cur.todo;
        if (todo == 0) return;
        int m = (l + r) / 2;
        apply(2 * o, l, m, todo);
        apply(2 * o + 1, m + 1, r, todo);
        cur.todo = TODO_INIT;
    }

    void build(const vector<T>& a, int o, int l, int r) {
        t[o].todo = TODO_INIT;
        if (l == r) {
            t[o].val = a[l];
            return;
        }
        int m = (l + r) / 2;
        build(a, 2 * o, l, m);
        build(a, 2 * o, m + 1, r);
        maintain(o);
    }

    void update(int o, int l, int r, int ql, int qr, F todo) {
        if (l >= ql && r <= qr) {
            apply(o, l, r, todo);
            return;
        }
        spread(o, l, r);
        int m = (l + r) / 2;
        if (ql <= m) {
            update(2 * o, l, m, ql, qr, todo);
        }
        if (qr > m) {
            update(2 * o + 1, m + 1, r, ql, qr, todo);
        }
        maintain(o);
    }

    int find_first(int o, int l, int r, int ql, int qr, int target) {
        if (r < ql || l > qr || target < t[o].val.first || target > t[o].val.second) {
            return -1;
        }
        if (l == r) {
            return l;
        }
        spread(o, l, r);
        int m = (l + r) / 2;
        int idx = find_first(2 * o, l, m, ql, qr, target);
        if (idx < 0) {
            idx = find_first(2 * o + 1, m + 1, r, ql, qr, target);
        }
        return idx;
    }
public:
    lazy_seg(int n, T init_val = { 0, 0 }) : lazy_seg(vector<T>(n, init_val)) {}
    lazy_seg(const vector<T>& a) : n(a.size()), t(2 << bit_width(a.size() - 1)) {
        build(a, 1, 0, n - 1);
    }

    void update(int l, int r, F f) {
        update(1, 0, n - 1, l, r, f);
    }

    int find_first(int l, int r, int target) {
        return find_first(1, 0, n - 1, l, r, target);
    }
};

// 可持久化线段树
class node {
    int l, r;
    node* lo;
    node* ro;
    int cnt;
    void maintain() {
        cnt = lo->cnt + ro->cnt;
        sum = lo->sum + ro->sum;
    }
public:
    long long sum;
    node(int l, int r, node* lo = nullptr, node* ro = nullptr, long long cnt = 0, long long sum = 0)
        : l(l), r(r), lo(lo), ro(ro), cnt(cnt), sum(sum) {
    }

    static node* build(int l, int r) {
        node* o = new node(l, r);
        if (l == r) return o;
        int mid = (l + r) / 2;
        o->lo = build(l, mid);
        o->ro = build(mid + 1, r);
        return o;
    }

    node* add(int i, int x) {
        node* o = new node(l, r, lo, ro, cnt, sum);
        if (l == r) {
            o->cnt++;
            o->sum += x;
            return o;
        }
        int mid = (l + r) / 2;
        if (i <= mid) {
            o->lo = o->lo->add(i, x);
        }
        else {
            o->ro = o->ro->add(i, x);
        }
        o->maintain();
        return o;
    }

    int kth(node* old, int k) {
        if (l == r) return l;
        int cnt_l = lo->cnt - old->lo->cnt;
        if (cnt_l >= k) {
            return lo->kth(old->lo, k);
        }
        return ro->kth(old->ro, k - cnt_l);
    }

    pair<int, long long> query(node* old, int i) {
        if (r <= i) {
            return { cnt - old->cnt, sum - old->sum };
        }
        auto [cnt, sum] = lo->query(old->lo, i);
        int mid = (l + r) / 2;
        if (i > mid) {
            auto [c, s] = ro->query(old->ro, i);
            cnt += c;
            sum += s;
        }
        return { cnt, sum };
    }
};

class Solution {
public:
    vector<long long> minOperations(vector<int>& nums, int k, vector<vector<int>>& queries) {
        int n = nums.size();
        vector<int> f(n);
        for (int i = 1; i < n; i++) {
            f[i] = nums[i] % k == nums[i - 1] % k ? f[i - 1] : i;
        }

        vector<int> sorted_nums = nums;
        ranges::sort(sorted_nums);
        sorted_nums.erase(ranges::unique(sorted_nums).begin(), sorted_nums.end());
        int m = sorted_nums.size();

        vector<node*> t(n + 1);
        t[0] = node::build(0, m);
        for (int i = 0; i < n; i++) {
            int j = ranges::lower_bound(sorted_nums, nums[i]) - sorted_nums.begin();
            t[i + 1] = t[i]->add(j, nums[i]);
        }

        vector<long long> ans;
        ans.reserve(queries.size());
        for (auto& q : queries) {
            int l = q[0], r = q[1];
            if (f[r] > f[l]) {
                ans.push_back(-1);
                continue;
            }
            r++;
            int m = r - l;
            int i = t[r]->kth(t[l], m / 2 + 1);
            long long midian = sorted_nums[i];

            long long tot = t[r]->sum - t[l]->sum;
            auto [cnt_l, sum_l] = t[r]->query(t[l], i);
            long long sum = midian * cnt_l - sum_l;
            sum += tot - sum_l - (m - cnt_l) * midian;
            ans.push_back(sum / k);
        }
        return ans;
    }
};

template<typename T, typename Compare = less<T>>
class lazy_heap {
    priority_queue<T, vector<T>, Compare> pq;
    unordered_map<T, int> remove_cnt;
    size_t sz = 0;

    void apply_remove() {
        while (!pq.empty() && remove_cnt[pq.top()] > 0) {
            remove_cnt[pq.top()]--;
            pq.pop();
        }
    }
public:
    size_t size() {
        return sz;
    }

    void remove(T x) {
        remove_cnt[x]++;
        sz--;
    }

    T top() {
        apply_remove();
        return pq.top();
    }

    T pop() {
        apply_remove();
        sz--;
        T x = pq.top();
        pq.pop();
        return x;
    }

    void push(T x) {
        if (remove_cnt[x] > 0) {
            remove_cnt[x]--;
        }
        else {
            pq.push(x);
        }
        sz++;
    }
};

// this is a eular's of mod 10, there had qpow in this region
namespace eular_10 {
    const int mod = 10;
    const int mx = 1001;
    int pow2[4] = { 2, 4, 8, 6 };

    int f[mx]{};
    int inv_f[mx]{};
    int p2[mx]{};
    int p5[mx]{};

    int qpow(int x, int n) {
        int res = 1;
        while (n) {
            if (n & 1) {
                res = res * x % mod;
            }
            x = x * x % mod;
            n /= 2;
        }
        return res;
    }

    auto init = []() {
        f[0] = inv_f[0] = 1;
        for (int i = 1; i < mx; i++) {
            int x = i;

            int e2 = countr_zero(unsigned(x));
            x >>= e2;
            int e5 = 0;
            while (x % 5 == 0) {
                e5++; x /= 5;
            }
            f[i] = f[i - 1] * x % mod;
            inv_f[i] = qpow(f[i], 3);
            p2[i] = p2[i - 1] + e2;
            p5[i] = p5[i - 1] + e5;
        }
        return  0;
        }();

    int comb(int n, int k) {
        int e2 = p2[n] - p2[n - k] - p2[k];
        return f[n] * inv_f[n - k] * inv_f[k] *
            (e2 ? pow2[(e2 - 1) % 4] : 1) * //因为数组下标从0开始，所以这里需要减一
            (p5[n] - p5[n - k] - p5[k] ? 5 : 1) % mod;
    }
}

// template of matrix 
namespace matrix_func {
    using matrix = vector<vector<long long>>;
    constexpr int mod = 1e9 + 7;

    matrix mul(matrix& a, matrix& b) { // multiply two matrices
        int n = a.size(), m = b[0].size();
        matrix c(n, vector<long long>(m));
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < a[i].size(); k++) { // this is enumerating columns of matrix a
                if (a[i][k] == 0) continue;
                for (int j = 0; j < m; j++) { // this operation is multiply a[i][k] and all number of kth row of matrix b
                    c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % mod; // because row of matrix b should be equal to column of matrix a
                }
            }
        }
        return c;
    }

    matrix pow(matrix a, int n) {
        int m = a.size();
        matrix res;
        for (int i = 0; i < m; i++) res[i][i] = 1; // identity matrix
        while (n) {
            if (n & 1) {
                res = mul(a, res);
            }
            a = mul(a, a);
            n /= 2;
        }
        return res;
    } //  matrix a should be an identity matrix

    matrix pow(matrix a, long long n) {
        int m = a.size();
        matrix res;
        for (int i = 0; i < m; i++) res[i][i] = 1; // identity matrix
        while (n) {
            if (n & 1) {
                res = mul(a, res);
            }
            a = mul(a, a);
            n /= 2;
        }
        return res;
    } //  matrix a should be an identity matrix

    matrix pow_mul(matrix a, int n, matrix f0) {
        matrix res = f0;
        while (n) {
            if (n & 1) {
                res = mul(a, res);
            }
            a = mul(a, a);
            n /= 2;
        }
        return res;
    }

    matrix pow_mul(matrix a, long long n, matrix f0) { // if n is bigger, you need to use long long
        matrix res = f0;
        while (n) {
            if (n & 1) {
                res = mul(a, res);
            }
            a = mul(a, a);
            n /= 2;
        }
        return res;
    }
}

namespace Fact {
    using ll = long long;
    const int mod = 1e9 + 7;
    const int mx = 31; // 根据需要修改
    int f[mx];
    int inv_f[mx];

    int pow(ll x, int n) {
        ll res = 1;
        while (n) {
            if (n & 1) {
                res = res * x % mod;
            }
            x = x * x % mod;
            n /= 2;
        }
        return res;
    }

    auto init = []() {
        f[0] = inv_f[0] = 1;
        for (int i = 1; i < mx; i++) {
            f[i] = 1ll * i * f[i - 1] % mod;
        }
        inv_f[mx - 1] = pow(f[mx - 1], mod - 2);
        for (int i = mx - 1; i > 1; i--) {
            inv_f[i - 1] = 1ll * inv_f[i] * i % mod;
        }
        return  0;
        }();
}

// pretreat remaining square core
namespace rsc {
    const int mx = 1e5 + 1;
    int core[mx];
    auto init = [] {
        for (int i = 1; i < mx; i++) {
            if (!core[i]) {
                for (int j = 1; i * j * j < mx; j++) {
                    core[i * j * j] = i;
                }
            }
        }
        return 0;
        }();
}

long long qpow(long long x, long long n) {
    long long res = 1;
    while (n) {
        if (n & 1) {
            res = res * x % mod;
        }
        x = x * x % mod;
        n /= 2;
    }
    return res;
}

