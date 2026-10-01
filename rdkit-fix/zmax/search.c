/* Local search for the connected 4-regular graph with the most matchings, at sizes beyond
 * exhaustive enumeration.
 *
 * Exhaustive generation dies around n=16 (8.0M graphs) and exact subset DP runs out of memory
 * around n=24 (2^24 states). Between those, the maximum can still be attacked by local search
 * with an EXACT objective: start from a circulant (4-regular and connected), and apply 2-opt
 * edge swaps -- replace (a,b),(c,d) with (a,c),(b,d) -- which preserve every degree.
 *
 * The result is a rigorous LOWER bound on Z_max, not a proof of optimality. Its credibility is
 * established by running the same search at n <= 15, where the true maximum is known from geng,
 * and checking that it is recovered.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned long long *f;
static int N;
static unsigned int adj[32];

static unsigned long long hosoya(void) {
  const unsigned int full = (1u << N) - 1u;
  f[0] = 1;
  for (unsigned int S = 1; S <= full; S++) {
    int v = __builtin_ctz(S);
    unsigned int S2 = S & ~(1u << v);
    unsigned long long t = f[S2];
    unsigned int nb = adj[v] & S2;
    while (nb) { int u = __builtin_ctz(nb); t += f[S2 & ~(1u << u)]; nb &= nb - 1; }
    f[S] = t;
  }
  return f[full];
}

static int connected(void) {
  unsigned int seen = 1u, frontier = 1u;
  while (frontier) {
    unsigned int next = 0;
    while (frontier) {
      int v = __builtin_ctz(frontier); frontier &= frontier - 1;
      next |= adj[v] & ~seen;
    }
    seen |= next; frontier = next;
  }
  return seen == ((1u << N) - 1u);
}

static void add(int a, int b) { adj[a] |= 1u << b; adj[b] |= 1u << a; }
static void del(int a, int b) { adj[a] &= ~(1u << b); adj[b] &= ~(1u << a); }
static int has(int a, int b) { return (adj[a] >> b) & 1; }

int main(int argc, char **argv) {
  N = atoi(argv[1]);
  long iters = argc > 2 ? atol(argv[2]) : 200000;
  unsigned seed = argc > 3 ? (unsigned)atoi(argv[3]) : 1;
  srandom(seed);
  f = malloc(sizeof(unsigned long long) << N);
  if (!f) { fprintf(stderr, "oom at n=%d\n", N); return 1; }

  /* circulant C_N(1,2): 4-regular and connected for N >= 5 */
  for (int i = 0; i < N; i++) adj[i] = 0;
  for (int i = 0; i < N; i++) { add(i, (i + 1) % N); add(i, (i + 2) % N); }

  unsigned long long best = hosoya();
  unsigned int bestadj[32]; memcpy(bestadj, adj, sizeof adj);

  for (long it = 0; it < iters; it++) {
    int a = random() % N, c = random() % N;
    unsigned int na = adj[a], nc = adj[c];
    if (!na || !nc) continue;
    int bi = random() % __builtin_popcount(na), di = random() % __builtin_popcount(nc);
    int b = -1, d = -1;
    for (unsigned int m = na; m; m &= m - 1) if (bi-- == 0) { b = __builtin_ctz(m); break; }
    for (unsigned int m = nc; m; m &= m - 1) if (di-- == 0) { d = __builtin_ctz(m); break; }
    if (b < 0 || d < 0) continue;
    if (a == c || a == d || b == c || b == d) continue;
    if (has(a, c) || has(b, d)) continue;

    del(a, b); del(c, d); add(a, c); add(b, d);
    if (!connected()) { del(a, c); del(b, d); add(a, b); add(c, d); continue; }
    unsigned long long z = hosoya();
    if (z >= best) {                       /* accept ties, to drift across plateaus */
      if (z > best) { best = z; memcpy(bestadj, adj, sizeof adj); }
    } else {
      del(a, c); del(b, d); add(a, b); add(c, d);
    }
  }
  printf("%d %llu %.6f\n", N, best, log2((double)best) / N);
  free(f);
  return 0;
}
