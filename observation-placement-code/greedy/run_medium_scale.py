#!/usr/bin/env python3
"""Medium-scale comparison of greedy information and greedy accuracy placement.

One self-contained file: the C++ engine is embedded below and compiled on first use.

  python3 run_medium_scale.py --test     compile, check against exact channels, measure speed
  python3 run_medium_scale.py            run everything; safe to stop and restart
  python3 run_medium_scale.py --status   show progress
  python3 run_medium_scale.py --package  build the results zip again

Needs Python 3.8+, g++ or clang++, and networkx (pip install networkx). Results go to
medium_scale_run/ next to this file. The finished zip is the only thing to send back.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse, bisect, csv, gzip, hashlib, itertools, json, math, os, platform, shutil, subprocess, sys, tempfile, time, zipfile

RUNNER_VERSION = 2
R_FORMAL = 1 << 16
SEED = 810000
GRAPHS = 10
FAMILIES = ['ER', 'BA', 'WS', 'REG']
NU = [1.]
RHO = [.1, .35]
RANDOMS = 32
BUDGETS = {20: [2, 4, 6, 10], 50: [3, 5, 10, 15, 25], 100: [5, 10, 20, 30, 50], 200: [10, 20, 40]}
EXCHANGE_MMAX = {20: 0, 50: 0, 100: 0, 200: 0}
SENSITIVITY = {50}
# Rough single-process hours per graph and peak GiB per process, used for planning only.
HOURS = {20: .01, 50: .08, 100: .2, 200: .53}
PEAK_GIB = {20: .2, 50: .7, 100: 1.4, 200: 3.}
DISK_GIB = {20: .05, 50: .4, 100: 1.2, 200: 4.2}
# Single-process benchmark seconds on the machine that produced the time estimates.
REFERENCE_SECONDS = 14.6
HERE = Path(__file__).resolve().parent
RUN = HERE / 'medium_scale_run'
RESULTS = RUN / 'results'
LOG = RUN / 'progress.log'
EXE = RUN / ('engine.exe' if os.name == 'nt' else 'engine')

ENGINE_SOURCE = r'''// All-source first-passage channels and observation-set selection.
//   calib  graph.txt out.csv realizations seed
//   gen    graph.txt prefix realizations seed tau1,tau2,...
//   eval   graph.txt select.bin train.bin test.bin R seed out.csv mmax budgets exchange_mmax randoms [fixed.txt|sens]
// Every undirected edge gets an independent Exp(1) length per realization, and
// u infects v by time tau iff their weighted distance is at most tau. Bit files
// hold, for each realization and node v, the N-bit set of sources reaching v.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using std::string; using std::vector;
using u64 = uint64_t; using u32 = uint32_t;

struct Graph {
  int n = 0, e = 0;
  vector<vector<std::pair<int,int>>> adj;
};

static Graph read_graph(const string &path) {
  std::ifstream f(path); Graph g; f >> g.n >> g.e; g.adj.resize(g.n);
  for (int j = 0; j < g.e; j++) { int u, v; f >> u >> v; g.adj[u].push_back({v,j}); g.adj[v].push_back({u,j}); }
  return g;
}

static void dijkstra(const Graph &g, const vector<double> &w, int start, double limit, vector<double> &d) {
  std::fill(d.begin(), d.end(), std::numeric_limits<double>::infinity()); d[start] = 0;
  using Item = std::pair<double,int>;
  std::priority_queue<Item, vector<Item>, std::greater<Item>> pq; pq.push({0.,start});
  while (!pq.empty()) {
    auto [du,u] = pq.top(); pq.pop(); if (du != d[u]) continue;
    for (auto [v,j] : g.adj[u]) { double dv = du + w[j]; if (dv <= limit && dv < d[v]) { d[v] = dv; pq.push({dv,v}); } }
  }
}

static int calib(int argc, char **argv) {
  Graph g = read_graph(argv[2]); int R = std::stoi(argv[4]); std::mt19937_64 rng(std::stoull(argv[5]));
  std::exponential_distribution<double> ex(1.);
  const int bins = 1200; const double lo = -5, hi = 3; // log10 tau grid
  vector<double> hist(bins + 1, 0.), w(g.e), d(g.n);
  for (int r = 0; r < R; r++) {
    for (double &x : w) x = ex(rng);
    for (int u = 0; u < g.n; u++) {
      dijkstra(g, w, u, std::numeric_limits<double>::infinity(), d);
      for (int v = 0; v < g.n; v++) if (v != u) {
        int b = (int)std::ceil((std::log10(d[v]) - lo) / (hi - lo) * bins);
        hist[std::clamp(b, 0, bins)] += 1;
      }
    }
  }
  std::ofstream out(argv[3]); out << "tau,mean_infected\n"; out.precision(12); double total = 0;
  for (int b = 0; b <= bins; b++) { total += hist[b]; out << std::pow(10., lo + (hi - lo) * b / bins) << ',' << 1 + total / (double(R) * g.n) << '\n'; }
  return 0;
}

static int gen(int argc, char **argv) {
  Graph g = read_graph(argv[2]); string prefix = argv[3]; int R = std::stoi(argv[4]);
  std::mt19937_64 rng(std::stoull(argv[5])); std::exponential_distribution<double> ex(1.);
  vector<double> taus; { std::stringstream ss(argv[6]); string x; while (std::getline(ss, x, ',')) taus.push_back(std::stod(x)); }
  double limit = *std::max_element(taus.begin(), taus.end());
  int W = (g.n + 63) / 64; vector<std::ofstream> out;
  for (size_t t = 0; t < taus.size(); t++) out.emplace_back(prefix + "_t" + std::to_string(t) + ".bin", std::ios::binary);
  vector<double> w(g.e), d(g.n); vector<vector<u64>> rows(taus.size(), vector<u64>((size_t)W * g.n));
  for (int r = 0; r < R; r++) {
    for (double &x : w) x = ex(rng);
    for (auto &m : rows) std::fill(m.begin(), m.end(), 0);
    for (int v = 0; v < g.n; v++) {
      dijkstra(g, w, v, limit, d);
      for (int u = 0; u < g.n; u++) if (d[u] <= limit)
        for (size_t t = 0; t < taus.size(); t++) if (d[u] <= taus[t]) rows[t][(size_t)v * W + u / 64] |= 1ULL << (u % 64);
    }
    for (size_t t = 0; t < taus.size(); t++) out[t].write((char*)rows[t].data(), rows[t].size() * sizeof(u64));
  }
  return 0;
}

struct Bits {
  vector<u64> data; const u64 *p = nullptr; int n, W; long R;
  Bits(const string &path, int n_, long R_) : n(n_), W((n_ + 63) / 64), R(R_) {
    std::ifstream f(path, std::ios::binary | std::ios::ate); size_t bytes = (size_t)R * n * W * 8;
    if (!f || (size_t)f.tellg() != bytes) { std::cerr << "bad bit file " << path << "\n"; exit(3); }
    data.resize(bytes / 8); f.seekg(0); f.read((char*)data.data(), bytes);
    if (!f) { std::cerr << "cannot read " << path << "\n"; exit(3); }
    p = data.data();
  }
  const u64 *row(long r, int v) const { return p + ((size_t)r * n + v) * W; }
};

template <class F> static inline void for_bits(const u64 *row, int W, F f) {
  for (int i = 0; i < W; i++) { u64 x = row[i]; while (x) { f(i * 64 + __builtin_ctzll(x)); x &= x - 1; } }
}

// Partition of (realization, source) pairs by the report on the current set.
struct Partition {
  const Bits &X; int N; long R; vector<long> rows; vector<u32> id, C, ystart, D, map0, map1; vector<double> xl; double plugin = 0;
  explicit Partition(const Bits &b) : X(b), N(b.n), R(b.R), rows(b.R), id((size_t)b.R * b.n) {
    std::iota(rows.begin(), rows.end(), 0L); setup();
  }
  // A partition restricted to the given realizations, for delete-one-batch estimates.
  Partition(const Bits &b, vector<long> kept) : X(b), N(b.n), R(kept.size()), rows(std::move(kept)), id((size_t)R * b.n) { setup(); }
  void setup() {
    xl.resize(R + 1); xl[0] = 0; for (long c = 1; c <= R; c++) xl[c] = c * std::log((double)c);
    reset();
  }
  void reset() {
    for (long r = 0; r < R; r++) for (int u = 0; u < N; u++) id[(size_t)r * N + u] = u;
    C.assign(N, (u32)R); ystart = {0, (u32)N}; D.assign(N, 0);
  }
  void count(int v) {
    D.assign(C.size(), 0);
    for (long r = 0; r < R; r++) { const u32 *ids = &id[(size_t)r * N]; for_bits(X.row(rows[r], v), X.W, [&](int u) { D[ids[u]]++; }); }
  }
  // Scores of refining by v. acc is the plug-in count sum of per-report maxima.
  void score(int v, long &acc, double &info) {
    count(v);
    const double RN = (double)R * N; long a = 0; double sumxl = 0, sumny = 0; long K = 0, E = 0;
    for (size_t y = 0; y + 1 < ystart.size(); y++) {
      u32 m0 = 0, m1 = 0; long n0 = 0, n1 = 0;
      for (u32 j = ystart[y]; j < ystart[y + 1]; j++) {
        u32 d = D[j], c0 = C[j] - d; n1 += d; n0 += c0; m1 = std::max(m1, d); m0 = std::max(m0, c0);
        sumxl += xl[d] + xl[c0]; E += (d > 0) + (c0 > 0);
      }
      a += m0 + m1;
      if (n0) { sumny += n0 * std::log((double)n0); K++; }
      if (n1) { sumny += n1 * std::log((double)n1); K++; }
    }
    acc = a;
    double Hy = std::log(RN) - sumny / RN, HyU = std::log((double)R) - sumxl / RN;
    plugin = Hy - HyU;
    info = plugin + ((K - 1) - (E - N)) / (2 * RN); // Miller--Madow for both entropies
  }
  void refine(int v) {
    count(v);
    size_t E = C.size(); map0.assign(E, 0); map1.assign(E, 0);
    vector<u32> nc, ns; nc.reserve(E + 16); ns.push_back(0);
    for (size_t y = 0; y + 1 < ystart.size(); y++) {
      bool any = false;
      for (u32 j = ystart[y]; j < ystart[y + 1]; j++) if (C[j] > D[j]) { map0[j] = nc.size(); nc.push_back(C[j] - D[j]); any = true; }
      if (any) ns.push_back(nc.size());
      any = false;
      for (u32 j = ystart[y]; j < ystart[y + 1]; j++) if (D[j]) { map1[j] = nc.size(); nc.push_back(D[j]); any = true; }
      if (any) ns.push_back(nc.size());
    }
    vector<u64> mask(X.W);
    for (long r = 0; r < R; r++) {
      const u64 *row = X.row(rows[r], v); u32 *ids = &id[(size_t)r * N];
      for (int u = 0; u < N; u++) ids[u] = (row[u / 64] >> (u % 64) & 1) ? map1[ids[u]] : map0[ids[u]];
    }
    C.swap(nc); ystart.swap(ns); D.assign(C.size(), 0);
  }
};

enum Crit { INFO, ACC, PLUGIN };

static int pick(const vector<int> &cand, const vector<long> &acc, const vector<double> &info, Crit c, std::mt19937_64 &rng) {
  vector<int> tied;
  if (c == ACC) { long b = *std::max_element(acc.begin(), acc.end()); for (size_t i = 0; i < cand.size(); i++) if (acc[i] == b) tied.push_back(i); }
  else { double b = *std::max_element(info.begin(), info.end()); for (size_t i = 0; i < cand.size(); i++) if (info[i] >= b - 1e-11) tied.push_back(i); }
  return cand[tied[std::uniform_int_distribution<size_t>(0, tied.size() - 1)(rng)]];
}

static vector<int> greedy(Partition &P, int mmax, Crit c, std::mt19937_64 &rng) {
  P.reset(); vector<int> path; vector<char> used(P.N, 0);
  for (int k = 0; k < mmax; k++) {
    vector<int> cand; for (int v = 0; v < P.N; v++) if (!used[v]) cand.push_back(v);
    vector<long> a(cand.size()); vector<double> s(cand.size());
    for (size_t i = 0; i < cand.size(); i++) { P.score(cand[i], a[i], s[i]); if (c == PLUGIN) s[i] = P.plugin; }
    int v = pick(cand, a, s, c, rng); P.refine(v); used[v] = 1; path.push_back(v);
  }
  return path;
}

// Greedy on the delete-one-batch jackknife of plug-in information, with 8 batches.
static vector<int> greedy_jack(const Bits &S, int mmax, std::mt19937_64 &rng) {
  Partition P(S); vector<Partition> sub; long R = S.R;
  for (int b = 0; b < 8; b++) { vector<long> kept; for (long r = 0; r < R; r++) if (r * 8 / R != b) kept.push_back(r); sub.emplace_back(S, kept); }
  vector<int> path; vector<char> used(P.N, 0);
  for (int k = 0; k < mmax; k++) {
    vector<int> cand; for (int v = 0; v < P.N; v++) if (!used[v]) cand.push_back(v);
    vector<long> a(cand.size()); vector<double> s(cand.size());
    for (size_t i = 0; i < cand.size(); i++) {
      double x, mean = 0; long y; P.score(cand[i], a[i], x); double full = P.plugin;
      for (auto &q : sub) { q.score(cand[i], y, x); mean += q.plugin / 8; }
      s[i] = 8 * full - 7 * mean;
    }
    int v = pick(cand, a, s, INFO, rng); P.refine(v); for (auto &q : sub) q.refine(v); used[v] = 1; path.push_back(v);
  }
  return path;
}

// Steepest 1-exchange ascent on the selection sample, at most max_pass passes.
static vector<int> exchange(Partition &P, vector<int> M, Crit c, int max_pass) {
  int m = M.size();
  auto value = [&](long a, double s) { return c == ACC ? (double)a : s; };
  for (int pass = 0; pass < max_pass; pass++) {
    double current = -1e300, best = -1e300; int ba = -1, bb = -1;
    for (int ia = 0; ia < m; ia++) {
      P.reset(); for (int j = 0; j < m; j++) if (j != ia) P.refine(M[j]);
      std::set<int> rest(M.begin(), M.end()); rest.erase(M[ia]);
      for (int b = 0; b < P.N; b++) if (!rest.count(b)) {
        long a; double s; P.score(b, a, s); double x = value(a, s);
        if (b == M[ia]) current = x;
        if (b != M[ia] && x > best) { best = x; ba = ia; bb = b; }
      }
    }
    if (best <= current + (c == ACC ? 0.5 : 1e-11)) break;
    M[ba] = bb;
  }
  return M;
}

struct Eval { double a_test, a_plugin, unseen, info, batch[8]; long patterns; bool extra; double top1, top3, top5, rank, dist, eff; };

// Open-addressing table from 64-bit keys to counts.
struct Table {
  vector<u64> key; vector<u32> val; size_t mask, used = 0;
  explicit Table(size_t cap = 1 << 16) : key(cap, ~0ULL), val(cap, 0), mask(cap - 1) {}
  static inline size_t mix(u64 x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; return x; }
  u32 &at(u64 k) {
    if ((used + 1) * 10 > (mask + 1) * 7) grow();
    size_t i = mix(k) & mask;
    while (key[i] != k) { if (key[i] == ~0ULL) { key[i] = k; used++; break; } i = (i + 1) & mask; }
    return val[i];
  }
  const u32 *find(u64 k) const {
    size_t i = mix(k) & mask;
    while (key[i] != k) { if (key[i] == ~0ULL) return nullptr; i = (i + 1) & mask; }
    return &val[i];
  }
  void grow() {
    Table t((mask + 1) * 2);
    for (size_t i = 0; i <= mask; i++) if (key[i] != ~0ULL) t.at(key[i]) = val[i];
    key.swap(t.key); val.swap(t.val); mask = t.mask; used = t.used;
  }
};

static inline void fill_patterns(const Bits &X, long r, const vector<int> &M, vector<u64> &pat) {
  std::fill(pat.begin(), pat.end(), 0);
  for (size_t k = 0; k < M.size(); k++) for_bits(X.row(r, M[k]), X.W, [&](int u) { pat[u] |= 1ULL << k; });
}

// Hit probabilities and ranks break ties uniformly, so an unseen report gives k/N and
// (N+1)/2. Error distance uses the MAP rule, and the mean hop distance for an unseen report.
static Eval evaluate(const Bits &train, const Bits &test, const vector<int> &M, const vector<vector<int>> *hop = nullptr) {
  int N = train.n; long R = train.R; vector<u64> pat(N); Table pairs(1 << 18);
  if (M.size() > 54 || N > 512) { std::cerr << "pattern key overflow\n"; exit(4); }
  for (long r = 0; r < R; r++) { fill_patterns(train, r, M, pat); for (int u = 0; u < N; u++) pairs.at(pat[u] << 9 | (u64)u)++; }
  // For each report keep its maximum count and the corresponding source.
  std::map<u64,int> dummy; Table best(1 << 16), who(1 << 16), total(1 << 16);
  double sumxl = 0; long E = 0;
  for (size_t i = 0; i <= pairs.mask; i++) if (pairs.key[i] != ~0ULL) {
    u64 y = pairs.key[i] >> 9; u32 c = pairs.val[i], u = pairs.key[i] & 511; E++; sumxl += c * std::log((double)c);
    u32 &b = best.at(y); if (c > b || (c == b && u + 1 < who.at(y))) { b = c; who.at(y) = u + 1; }
    total.at(y) += c;
  }
  const double RN = (double)R * N; Eval out{}; long plug = 0; double sumny = 0; long K = 0;
  for (size_t i = 0; i <= best.mask; i++) if (best.key[i] != ~0ULL) plug += best.val[i];
  for (size_t i = 0; i <= total.mask; i++) if (total.key[i] != ~0ULL) { sumny += total.val[i] * std::log((double)total.val[i]); K++; }
  out.a_plugin = plug / RN; out.patterns = K;
  out.info = (std::log(RN) - sumny / RN) - (std::log((double)R) - sumxl / RN) + ((K - 1) - (E - N)) / (2 * RN);
  long Rt = test.R; double correct = 0, unseen = 0, b[8] = {0};
  for (long r = 0; r < Rt; r++) {
    fill_patterns(test, r, M, pat); double got = 0;
    for (int u = 0; u < N; u++) {
      const u32 *w = who.find(pat[u]);
      if (!w) { unseen += 1; got += 1. / N; } else if ((int)*w - 1 == u) got += 1;
    }
    correct += got; b[r * 8 / Rt] += got;
  }
  out.a_test = correct / ((double)Rt * N); out.unseen = unseen / ((double)Rt * N);
  for (int i = 0; i < 8; i++) out.batch[i] = b[i] / ((double)N * ((i + 1) * Rt / 8 - i * Rt / 8));
  if (!hop) return out;
  vector<std::pair<u64,u32>> rows; rows.reserve(pairs.used);
  for (size_t i = 0; i <= pairs.mask; i++) if (pairs.key[i] != ~0ULL) rows.push_back({pairs.key[i] >> 9, pairs.val[i]});
  std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &c) { return a.first != c.first ? a.first < c.first : a.second > c.second; });
  Table start(1 << 16); double eff = 0; vector<size_t> edge;
  for (size_t i = 0; i < rows.size();) {
    size_t j = i; double t = 0, sum = 0;
    while (j < rows.size() && rows[j].first == rows[i].first) { t += rows[j].second; sum += rows[j].second * std::log((double)rows[j].second); j++; }
    start.at(rows[i].first) = (u32)edge.size() + 1; edge.push_back(i); eff += t / RN * std::exp(std::log(t) - sum / t); i = j;
  }
  edge.push_back(rows.size());
  vector<double> far(N, 0.); for (int u = 0; u < N; u++) for (int w = 0; w < N; w++) far[u] += (double)(*hop)[w][u] / N;
  const int K3[3] = {1, 3, 5}; double hit[3] = {0, 0, 0}, rank = 0, dist = 0;
  for (long r = 0; r < Rt; r++) {
    fill_patterns(test, r, M, pat);
    for (int u = 0; u < N; u++) {
      const u32 *s = start.find(pat[u]); double above = 0, tied = N;
      if (s) {
        size_t a = edge[*s - 1], z = edge[*s]; const u32 *found = pairs.find(pat[u] << 9 | (u64)u); u32 c = found ? *found : 0;
        auto lo = std::partition_point(rows.begin() + a, rows.begin() + z, [&](const auto &x) { return x.second > c; });
        auto hi = std::partition_point(lo, rows.begin() + z, [&](const auto &x) { return x.second == c; });
        above = lo - (rows.begin() + a); tied = c ? hi - lo : N - (double)(z - a);
        dist += (*hop)[(int)*who.find(pat[u]) - 1][u];
      } else dist += far[u];
      rank += above + (tied + 1) / 2;
      for (int k = 0; k < 3; k++) hit[k] += std::clamp((K3[k] - above) / tied, 0., 1.);
    }
  }
  const double TN = (double)Rt * N;
  out.extra = true; out.top1 = hit[0] / TN; out.top3 = hit[1] / TN; out.top5 = hit[2] / TN; out.rank = rank / TN; out.dist = dist / TN; out.eff = eff;
  return out;
}

static vector<int> heuristic(const Graph &g, const string &kind, int mmax) {
  int n = g.n; vector<int> path;
  if (kind == "degree") {
    vector<int> order(n); std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return g.adj[a].size() > g.adj[b].size(); });
    return vector<int>(order.begin(), order.begin() + mmax);
  }
  if (kind == "coverage") {
    vector<char> covered(n, 0), used(n, 0);
    for (int k = 0; k < mmax; k++) {
      int best = -1, bv = -1;
      for (int v = 0; v < n; v++) if (!used[v]) { int c = !covered[v]; for (auto [w,j] : g.adj[v]) c += !covered[w]; if (c > bv) { bv = c; best = v; } }
      used[best] = 1; covered[best] = 1; for (auto [w,j] : g.adj[best]) covered[w] = 1; path.push_back(best);
    }
    return path;
  }
  // Farthest-first traversal: a 2-approximation to k-center, started at node 0.
  vector<int> dist(n, 1 << 30); int cur = 0;
  for (int k = 0; k < mmax; k++) {
    path.push_back(cur); std::queue<int> q; q.push(cur); dist[cur] = 0;
    while (!q.empty()) { int u = q.front(); q.pop(); for (auto [w,j] : g.adj[u]) if (dist[w] > dist[u] + 1) { dist[w] = dist[u] + 1; q.push(w); } }
    cur = std::max_element(dist.begin(), dist.end()) - dist.begin();
  }
  return path;
}

static int eval_cmd(int argc, char **argv) {
  Graph g = read_graph(argv[2]); long R = std::stol(argv[6]); std::mt19937_64 rng(std::stoull(argv[7]));
  Bits S(argv[3], g.n, R), T(argv[4], g.n, R), E(argv[5], g.n, R);
  std::ofstream out(argv[8]); out.precision(12);
  int mmax = std::stoi(argv[9]); vector<int> budgets; { std::stringstream ss(argv[10]); string x; while (std::getline(ss, x, ',')) budgets.push_back(std::stoi(x)); }
  int ex_mmax = std::stoi(argv[11]), randoms = std::stoi(argv[12]);
  out << "method,draw,m,set,a_test,a_plugin,unseen,info,patterns,seconds,b0,b1,b2,b3,b4,b5,b6,b7,top1,top3,top5,mean_rank,err_dist,eff_sources\n";
  auto wall = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  vector<vector<int>> hops(g.n, vector<int>(g.n, g.n));
  for (int s = 0; s < g.n; s++) {
    std::queue<int> q; q.push(s); hops[s][s] = 0;
    while (!q.empty()) { int u = q.front(); q.pop(); for (auto [w,j] : g.adj[u]) if (hops[s][w] > hops[s][u] + 1) { hops[s][w] = hops[s][u] + 1; q.push(w); } }
  }
  auto emit = [&](const string &method, int draw, const vector<int> &M, double secs, bool extra = true) {
    Eval e = evaluate(T, E, M, extra ? &hops : nullptr); out << method << ',' << draw << ',' << M.size() << ',';
    for (size_t i = 0; i < M.size(); i++) out << (i ? " " : "") << M[i];
    out << ',' << e.a_test << ',' << e.a_plugin << ',' << e.unseen << ',' << e.info << ',' << e.patterns << ',' << secs;
    for (double b : e.batch) out << ',' << b;
    if (e.extra) out << ',' << e.top1 << ',' << e.top3 << ',' << e.top5 << ',' << e.rank << ',' << e.dist << ',' << e.eff; else out << ",,,,,,";
    out << '\n'; out.flush();
  };
  bool sens = argc > 13 && string(argv[13]) == "sens";
  if (argc > 13 && !sens) { // Fixed sets only, for validation against exact channels.
    std::ifstream f(argv[13]); string line;
    while (std::getline(f, line)) { std::stringstream ss(line); vector<int> M; int v; while (ss >> v) M.push_back(v); emit("fixed", 0, M, 0); }
    return 0;
  }
  Partition P(S);
  for (auto [name, crit] : {std::pair<string,Crit>{"greedy_info", INFO}, {"greedy_acc", ACC}}) {
    double t0 = wall(); vector<int> path = greedy(P, mmax, crit, rng); double secs = wall() - t0;
    for (int m : budgets) {
      vector<int> M(path.begin(), path.begin() + m); emit(name, 0, M, secs);
      if (m <= ex_mmax) { double t1 = wall(); vector<int> L = exchange(P, M, crit, 5); emit(crit == ACC ? "exchange_acc" : "exchange_info", 0, L, wall() - t1); }
    }
  }
  for (string kind : {"degree", "coverage", "kcenter"}) {
    vector<int> path = heuristic(g, kind, mmax);
    for (int m : budgets) emit(kind, 0, vector<int>(path.begin(), path.begin() + m), 0);
  }
  vector<int> all(g.n); std::iota(all.begin(), all.end(), 0);
  for (int d = 0; d < randoms; d++) {
    std::shuffle(all.begin(), all.end(), rng);
    for (int m : budgets) emit("random", d, vector<int>(all.begin(), all.begin() + m), 0, d < 4);
  }
  if (sens) { // Selection-rule sensitivity; runs last so the other methods keep their random streams.
    double t0 = wall(); vector<int> path = greedy(P, mmax, PLUGIN, rng); double secs = wall() - t0;
    for (int m : budgets) emit("greedy_plugin", 0, vector<int>(path.begin(), path.begin() + m), secs);
    t0 = wall(); path = greedy_jack(S, mmax, rng); secs = wall() - t0;
    for (int m : budgets) emit("greedy_jack", 0, vector<int>(path.begin(), path.begin() + m), secs);
  }
  return 0;
}

int main(int argc, char **argv) {
  string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "calib" && argc == 6) return calib(argc, argv);
  if (cmd == "gen" && argc == 7) return gen(argc, argv);
  if (cmd == "eval" && argc >= 13) return eval_cmd(argc, argv);
  std::cerr << "usage: see header\n"; return 2;
}
'''


def say(text):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + text
    print(line, flush=True)
    RUN.mkdir(exist_ok=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def engine_hash():
    return hashlib.sha256(ENGINE_SOURCE.encode()).hexdigest()[:16]


def compile_engine():
    RUN.mkdir(exist_ok=True)
    stamp = RUN / 'engine.hash'
    if EXE.exists() and stamp.exists() and stamp.read_text().strip() == engine_hash():
        return
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        sys.exit('No C++ compiler found. On Ubuntu or WSL run: sudo apt install -y g++')
    source = RUN / 'medium_engine.cpp'
    source.write_text(ENGINE_SOURCE)
    say(f'compiling engine with {compiler}')
    done = subprocess.run([compiler, '-O3', '-std=c++17', str(source), '-o', str(EXE)], capture_output=True, text=True)
    if done.returncode:
        sys.exit('Compilation failed:\n' + done.stderr[-3000:])
    stamp.write_text(engine_hash())


def need_networkx():
    try:
        import networkx
        return networkx
    except ImportError:
        sys.exit('networkx is missing. Run: python3 -m pip install networkx')


def build(family, n, seed):
    nx = need_networkx()
    if family == 'ER':
        g = nx.gnp_random_graph(n, 4 / (n - 1), seed=seed)
    elif family == 'BA':
        g = nx.barabasi_albert_graph(n, 2, seed=seed)
    elif family == 'WS':
        g = nx.watts_strogatz_graph(n, 4, .1, seed=seed)
    elif family == 'REG':
        s = seed
        while True:
            g = nx.random_regular_graph(4, n, seed=s)
            if nx.is_connected(g):
                break
            s += 1
    else:
        raise ValueError(family)
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    return nx.convert_node_labels_to_integers(g, ordering='sorted')


def invert(curve_x, curve_y, target):
    """Linear interpolation of tau at a target mean infected count."""
    if target <= curve_x[0]:
        return curve_y[0]
    if target >= curve_x[-1]:
        return curve_y[-1]
    j = bisect.bisect_right(curve_x, target) - 1
    return curve_y[j] + (curve_y[j + 1] - curve_y[j]) / (curve_x[j + 1] - curve_x[j]) * (target - curve_x[j])


def stages(path, n, nus, rhos):
    """Invert the calibrated curve; coinciding targets are merged and recorded."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    taus = [float(r['tau']) for r in rows]
    means = [float(r['mean_infected']) for r in rows]
    out = []
    for kind, value, target in [('nu', x, 1 + x) for x in nus] + [('rho', x, x * n) for x in rhos]:
        if target <= 1.05 or target >= n:
            continue
        tau = invert(means, taus, target)
        for s in out:
            if abs(math.log(tau / s['tau'])) < .03:
                s['merged'].append([kind, value])
                break
        else:
            out.append(dict(kind=kind, target=value, tau=tau, merged=[]))
    return out


def run_graph(job):
    """Run one graph and write its results atomically; returns (label, seconds, error)."""
    label, start = job['label'], time.time()
    tmp = tempfile.mkdtemp(prefix=label + '_', dir=job['tmp'])
    try:
        g = build(job['family'], job['requested'], job['seed'])
        n = g.number_of_nodes()
        budgets = [m for m in job['budgets'] if m < n]
        graph = os.path.join(tmp, 'g.txt')
        with open(graph, 'w') as f:
            f.write(f'{n} {g.number_of_edges()}\n' + '\n'.join(f'{u} {v}' for u, v in g.edges) + '\n')
        curve = os.path.join(tmp, 'c.csv')
        subprocess.run([job['engine'], 'calib', graph, curve, '2048', str(job['seed'] + 11)], check=True)
        chosen = stages(curve, n, job['nu'], job['rho'])
        taus = ','.join(f'{s["tau"]:.10g}' for s in chosen)
        t0 = time.time()
        for k, sample in enumerate('STE'):
            subprocess.run([job['engine'], 'gen', graph, os.path.join(tmp, sample), str(job['R']), str(job['seed'] + job['mc_offset'] + 100 + k), taus], check=True)
        gen_seconds = time.time() - t0
        rows, timing, header = [], [], None
        for t, stage in enumerate(chosen):
            t1 = time.time()
            out = os.path.join(tmp, f'o{t}.csv')
            bits = [os.path.join(tmp, f'{sample}_t{t}.bin') for sample in 'STE']
            command = [job['engine'], 'eval', graph] + bits + [str(job['R']), str(job['seed'] + job['mc_offset'] + 200 + t), out,
                       str(max(budgets)), ','.join(map(str, budgets)), str(job['exchange']), str(job['randoms'])]
            if job['sensitivity']:
                command.append('sens')
            subprocess.run(command, check=True)
            with open(out, newline='') as f:
                reader = csv.reader(f)
                head = next(reader)
                header = head + ['graph', 'family', 'requested_n', 'n', 'stage_kind', 'stage_target', 'tau']
                extra = [label, job['family'], job['requested'], n, stage['kind'], stage['target'], stage['tau']]
                rows.extend(row + extra for row in reader)
            timing.append(dict(stage=f'{stage["kind"]}={stage["target"]}', eval_seconds=time.time() - t1))
            for path in bits:
                os.unlink(path)
        results = Path(job['out'])
        results.mkdir(parents=True, exist_ok=True)
        part = results / f'{label}.csv.gz.part'
        with gzip.open(part, 'wt', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(part, results / f'{label}.csv.gz')
        seconds = time.time() - start
        meta = dict(graph=label, family=job['family'], requested_n=job['requested'], seed=job['seed'], mc_offset=job['mc_offset'], n=n,
                    edges=g.number_of_edges(), edge_list=[list(e) for e in g.edges], R=job['R'], budgets=budgets, exchange_mmax=job['exchange'],
                    stages=chosen, gen_seconds=gen_seconds, total_seconds=seconds, timing=timing, engine=job['hash'])
        part = results / f'{label}.json.part'
        part.write_text(json.dumps(meta) + '\n')
        os.replace(part, results / f'{label}.json')
        return label, seconds, ''
    except Exception as error:
        return label, time.time() - start, f'{type(error).__name__}: {error}'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def all_jobs(tmp):
    """Deterministic grid, ordered so that a partial run stays balanced across families."""
    jobs = []
    sizes = sorted(BUDGETS)
    for i, requested in enumerate(sizes):
        for k in range(GRAPHS):
            for j, family in enumerate(FAMILIES):
                jobs.append(dict(label=f'{family}_{requested}_{k}', family=family, requested=requested, seed=SEED + 100000 * i + 10000 * j + 1000 * k,
                                 mc_offset=0, budgets=BUDGETS[requested], exchange=EXCHANGE_MMAX[requested], sensitivity=requested in SENSITIVITY,
                                 randoms=RANDOMS, nu=NU, rho=RHO, R=R_FORMAL, engine=str(EXE), tmp=str(tmp), out=str(RESULTS), hash=engine_hash()))
    return jobs


def finished():
    return {p.stem for p in RESULTS.glob('*.json')} if RESULTS.exists() else set()


def memory_gib():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30
    except (AttributeError, ValueError, OSError):
        pass
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(x, ctypes.c_ulonglong) for x in
                        ['total', 'avail', 'page', 'availpage', 'virt', 'availvirt', 'ext']]
        status = Status()
        status.length = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total / 2 ** 30
    except Exception:
        return 16.


def workers_for(n, chosen):
    """Keep about 6 GiB free for the system and stay within the logical cores."""
    cores = os.cpu_count() or 2
    by_memory = max(1, int((memory_gib() - 6) / PEAK_GIB[n]))
    return min(by_memory, chosen or cores, cores)


def status():
    jobs = all_jobs(RUN)
    done = finished()
    print(f'{len(done & {j["label"] for j in jobs})} of {len(jobs)} graphs finished')
    for n in sorted(BUDGETS):
        labels = [j['label'] for j in jobs if j['requested'] == n]
        print(f'  N={n}: {sum(x in done for x in labels)} of {len(labels)}')
    failures = RUN / 'failures.log'
    if failures.exists():
        print('Failures so far:\n' + failures.read_text()[-2000:])


def package():
    """Zip results, logs, and machine details; bit files never enter it."""
    target = RUN / 'medium_scale_results.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(RESULTS.glob('*')):
            if path.suffix in ('.gz', '.json'):
                z.write(path, 'results/' + path.name)
        for name in ['manifest.json', 'progress.log', 'failures.log', 'self_test.json', 'medium_engine.cpp']:
            if (RUN / name).exists():
                z.write(RUN / name, name)
    print(f'Results zip: {target} ({target.stat().st_size / 2 ** 20:.1f} MiB). Send this file back.')


def exact_channel(n, edges, tau):
    """Exact SI state law from each source by uniformization on the 2^n subsets."""
    neighbors = [set() for _ in range(n)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    moves = {}
    for x in range(1 << n):
        moves[x] = [(x | 1 << v, sum(x >> w & 1 for w in neighbors[v])) for v in range(n) if not x >> v & 1]
    rate = float(len(edges)) + 1
    laws = []
    for u in range(n):
        vector = {1 << u: 1.}
        total, weight, k, result = 0., math.exp(-rate * tau), 0, {}
        while total < 1 - 1e-13 and k < 2000:
            for x, p in vector.items():
                result[x] = result.get(x, 0.) + weight * p
            total += weight
            k += 1
            weight *= rate * tau / k
            following = {}
            for x, p in vector.items():
                leave = 0.
                for y, r in moves[x]:
                    if r:
                        following[y] = following.get(y, 0.) + p * r / rate
                        leave += r
                following[x] = following.get(x, 0.) + p * (1 - leave / rate)
            vector = following
        laws.append(result)
    return laws, neighbors


def exact_scores(n, neighbors, laws, M):
    channel = [{} for _ in range(n)]
    for u in range(n):
        for x, p in laws[u].items():
            y = tuple(x >> v & 1 for v in M)
            channel[u][y] = channel[u].get(y, 0.) + p
    reports = set(itertools.chain.from_iterable(channel))
    hops = []
    for s in range(n):
        d = {s: 0}
        queue = [s]
        for u in queue:
            for w in neighbors[u]:
                if w not in d:
                    d[w] = d[u] + 1
                    queue.append(w)
        hops.append(d)
    out = dict(a=0., info=0., top3=0., rank=0., dist=0.)
    for y in reports:
        row = sorted(((channel[u].get(y, 0.), u) for u in range(n)), key=lambda t: (-t[0], t[1]))
        py = sum(p for p, _ in row) / n
        out['a'] += row[0][0] / n
        for position, (p, u) in enumerate(row):
            out['top3'] += p / n if position < 3 else 0.
            out['rank'] += p / n * (position + 1)
            out['dist'] += p / n * hops[row[0][1]][u]
            if p > 0:
                out['info'] += p / n * math.log(p / py)
    return out


def self_test():
    """Compare engine estimates with exact values on a small asymmetric graph."""
    compile_engine()
    n, edges = 7, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4), (4, 5), (5, 6), (0, 2)]
    R = 1 << 15
    checks, failures = [], 0
    tmp = tempfile.mkdtemp(prefix='selftest_', dir=str(RUN))
    try:
        graph = os.path.join(tmp, 'g.txt')
        with open(graph, 'w') as f:
            f.write(f'{n} {len(edges)}\n' + '\n'.join(f'{a} {b}' for a, b in edges) + '\n')
        for t, tau in enumerate([.4, 1.2]):
            laws, neighbors = exact_channel(n, edges, tau)
            for k, sample in enumerate('STE'):
                subprocess.run([str(EXE), 'gen', graph, os.path.join(tmp, f'{sample}{t}'), str(R), str(4000 + 10 * t + k), str(tau)], check=True)
            out = os.path.join(tmp, f'o{t}.csv')
            subprocess.run([str(EXE), 'eval', graph] + [os.path.join(tmp, f'{s}{t}_t0.bin') for s in 'STE'] +
                           [str(R), str(5000 + t), out, '4', '1,2,3,4', '4', '2', 'sens'], check=True)
            with open(out) as f:
                for row in csv.DictReader(f):
                    M = list(map(int, row['set'].split()))
                    e = exact_scores(n, neighbors, laws, M)
                    pairs = [('a_test', e['a'], .012), ('info', e['info'], .015)]
                    if row['top3']:
                        pairs += [('top3', e['top3'], .012), ('mean_rank', e['rank'], .04), ('err_dist', e['dist'], .08)]
                    for name, truth, limit in pairs:
                        gap = float(row[name]) - truth
                        checks.append(dict(tau=tau, method=row['method'], set=row['set'], quantity=name, estimate=float(row[name]), exact=truth, gap=gap))
                        failures += abs(gap) > limit
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    worst = max(abs(c['gap']) for c in checks)
    methods = sorted({c['method'] for c in checks})
    (RUN / 'self_test.json').write_text(json.dumps(dict(failures=failures, count=len(checks), worst_gap=worst, methods=methods, checks=checks)) + '\n')
    say(f'self test: {len(checks)} comparisons with exact channels, {failures} failures, worst gap {worst:.4f}')
    return failures == 0


def benchmark():
    """Time one fixed graph and stage to scale the run-time estimate to this machine."""
    compile_engine()
    tmp = tempfile.mkdtemp(prefix='bench_', dir=str(RUN))
    try:
        job = dict(label='bench', family='ER', requested=100, seed=990000, mc_offset=0, budgets=[5, 10, 20], exchange=0, sensitivity=False, randoms=0,
                   nu=[], rho=[.1], R=1 << 15, engine=str(EXE), tmp=tmp, out=tmp, hash=engine_hash())
        label, seconds, error = run_graph(job)
        if error:
            sys.exit('Benchmark failed: ' + error)
        return seconds
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def estimate(seconds=None):
    plan = {n: workers_for(n, None) for n in BUDGETS}
    ratio = seconds / REFERENCE_SECONDS if seconds and REFERENCE_SECONDS else 1.
    cores = os.cpu_count() or 2
    hours = 0.
    for n, w in plan.items():
        # Logical cores beyond half are hyperthreads, worth about a quarter of a core.
        effective = min(w, cores / 2) + .25 * max(0, w - cores / 2) if cores >= 4 else w
        hours += HOURS[n] * GRAPHS * len(FAMILIES) * ratio / effective
    return plan, ratio, hours


def preflight(plan):
    free = shutil.disk_usage(str(RUN)).free / 2 ** 30
    need = max(DISK_GIB[n] * w for n, w in plan.items()) * 1.2
    say(f'memory {memory_gib():.0f} GiB, logical cores {os.cpu_count()}, free disk {free:.0f} GiB, peak temporary disk about {need:.0f} GiB')
    if free < need:
        sys.exit(f'Not enough disk: need about {need:.0f} GiB free in {RUN}. Move this file to a larger disk, then rerun.')


def run_all(chosen):
    compile_engine()
    if not (RUN / 'self_test.json').exists() and not self_test():
        sys.exit('Self test failed. Send medium_scale_run/self_test.json back before running further.')
    if json.loads((RUN / 'self_test.json').read_text())['failures']:
        sys.exit('The saved self test has failures. Send medium_scale_run/self_test.json back.')
    tmp = RUN / 'tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for stale in RESULTS.glob('*.part'):
        stale.unlink()
    plan = {n: workers_for(n, chosen) for n in BUDGETS}
    preflight(plan)
    manifest = RUN / 'manifest.json'
    info = dict(runner=RUNNER_VERSION, engine=engine_hash(), seed=SEED, graphs=GRAPHS, R=R_FORMAL, platform=platform.platform(), processor=platform.processor(),
                python=platform.python_version(), networkx=need_networkx().__version__, logical_cores=os.cpu_count(), memory_gib=memory_gib(), workers=plan)
    if manifest.exists():
        old = json.loads(manifest.read_text())
        for key in ['engine', 'seed', 'R', 'graphs']:
            if old.get(key) != info[key] and finished():
                sys.exit(f'Existing results used a different {key}. Keep them safe, remove medium_scale_run/results, then rerun.')
    manifest.write_text(json.dumps(info, indent=1) + '\n')
    done, jobs = finished(), all_jobs(tmp)
    remaining = [j for j in jobs if j['label'] not in done]
    say(f'{len(jobs) - len(remaining)} of {len(jobs)} graphs already finished; workers by size {plan}')
    failed = 0
    start = time.time()
    for n in sorted(BUDGETS):
        group = [j for j in remaining if j['requested'] == n]
        if not group:
            continue
        say(f'starting N={n}: {len(group)} graphs with {plan[n]} processes')
        with ProcessPoolExecutor(plan[n]) as pool:
            futures = [pool.submit(run_graph, j) for j in group]
            for count, future in enumerate(as_completed(futures), 1):
                label, seconds, error = future.result()
                if error:
                    failed += 1
                    with open(RUN / 'failures.log', 'a') as f:
                        f.write(f'{label}: {error}\n')
                    say(f'FAILED {label}: {error}')
                else:
                    say(f'N={n} {count}/{len(group)} {label} {seconds / 60:.1f} min; total elapsed {(time.time() - start) / 3600:.1f} h')
    shutil.rmtree(tmp, ignore_errors=True)
    missing = len(jobs) - len(finished() & {j['label'] for j in jobs})
    if missing:
        say(f'{missing} graphs not finished. Rerun this script to retry them; if failures repeat, try --workers with a smaller number.')
    else:
        say('all graphs finished')
    package()


def main():
    parser = argparse.ArgumentParser(description='Medium-scale placement comparison; safe to stop and restart.')
    parser.add_argument('--test', action='store_true', help='compile, check against exact channels, measure speed, then stop')
    parser.add_argument('--status', action='store_true', help='show progress')
    parser.add_argument('--package', action='store_true', help='rebuild the results zip')
    parser.add_argument('--workers', type=int, help='maximum processes; default chosen from cores and memory')
    a = parser.parse_args()
    RUN.mkdir(exist_ok=True)
    if a.status:
        return status()
    if a.package:
        return package()
    need_networkx()
    if a.test:
        compile_engine()
        if not self_test():
            sys.exit('Self test FAILED. Send medium_scale_run/self_test.json back before running further.')
        say('measuring speed (a few minutes)')
        seconds = benchmark()
        plan, ratio, hours = estimate(seconds)
        preflight(plan)
        say(f'benchmark {seconds:.0f} s, {ratio:.2f} times the reference; processes by size {plan}')
        say(f'estimated full run about {hours:.0f} h (range {.7 * hours:.0f} to {1.4 * hours:.0f} h)')
        say('test passed; start the full run with: python3 run_medium_scale.py')
        return
    run_all(a.workers)


if __name__ == '__main__':
    main()
