#include <bits/stdc++.h>
using namespace std;

static const int INF = 1e9;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    int sx, sy; cin >> sx >> sy; --sx; --sy;

    vector<string> grid(n);
    for (int i = 0; i < n; ++i) cin >> grid[i];

    string s; cin >> s;
    string t;
    t.reserve(s.size());
    for (char c : s) {
        if (t.empty() || t.back() != c) t.push_back(c);
    }

    const int N = n * m;
    auto id = [&](int i, int j){ return i * m + j; };

    array<vector<int>, 26> pos;
    for (int i = 0; i < 26; ++i) pos[i].clear();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            pos[grid[i][j] - 'a'].push_back(id(i,j));

    vector<int> A(N, INF), H(N, INF), D(N, INF);

    A[id(sx, sy)] = 0;
    vector<int> last_set = { id(sx, sy) };

    auto manhattan_transform = [&](void){
        for (int i = 0; i < n; ++i) {
            int base = i * m;
            int best = INF;
            for (int j = 0; j < m; ++j) {
                int idx = base + j;
                int v = A[idx];
                if (best + 1 < v) v = best + 1;
                H[idx] = v;
                best = v;
            }
            best = INF;
            for (int j = m - 1; j >= 0; --j) {
                int idx = base + j;
                int v = H[idx];
                if (best + 1 < v) {
                    v = best + 1;
                    H[idx] = v;
                }
                best = v;
            }
        }
        for (int j = 0; j < m; ++j) {
            int best = INF;
            for (int i = 0; i < n; ++i) {
                int idx = i * m + j;
                int v = H[idx];
                if (best + 1 < v) v = best + 1;
                D[idx] = v;
                best = v;
            }
            best = INF;
            for (int i = n - 1; i >= 0; --i) {
                int idx = i * m + j;
                int v = D[idx];
                if (best + 1 < v) {
                    v = best + 1;
                    D[idx] = v;
                }
                best = v;
            }
        }
    };

    int answer = 0;
    for (int step = 0; step < (int)t.size(); ++step) {
        manhattan_transform();

        for (int idx : last_set) A[idx] = INF;

        int letter = t[step] - 'a';
        const auto &cur = pos[letter];
        last_set = cur;

        if (step == (int)t.size() - 1) {
            int best = INF;
            for (int idx : cur) {
                int v = D[idx];
                A[idx] = v;
                if (v < best) best = v;
            }
            answer = best;
        } else {
            for (int idx : cur) {
                A[idx] = D[idx];
            }
        }
    }

    cout << answer << "\n";
    return 0;
}
