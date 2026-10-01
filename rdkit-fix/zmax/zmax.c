/* Read graph6 graphs on stdin, report the one with the most matchings.
 *
 * Z (the Hosoya index) is counted by subset DP: f[S] = matchings of the subgraph induced on S.
 * Taking v as the lowest set bit of S, either v is unmatched, or it is matched to one of its
 * neighbours in S. That is O(2^n * deg) per graph, which is what makes exhaustive enumeration
 * over geng's output affordable up to about n = 16. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long f[1u << 20];

static int parse_graph6(const char *s, unsigned int *adj) {
  int n = s[0] - 63;
  if (n < 0 || n > 20) return -1;
  for (int i = 0; i < n; i++) adj[i] = 0;
  const char *p = s + 1;
  int bit = 0, byte = *p++ - 63;
  for (int j = 1; j < n; j++) {
    for (int i = 0; i < j; i++) {
      if (bit == 6) { byte = *p++ - 63; bit = 0; }
      if ((byte >> (5 - bit)) & 1) { adj[i] |= 1u << j; adj[j] |= 1u << i; }
      bit++;
    }
  }
  return n;
}

static unsigned long long hosoya(const unsigned int *adj, int n) {
  const unsigned int full = (n == 32) ? 0xffffffffu : ((1u << n) - 1u);
  f[0] = 1;
  for (unsigned int S = 1; S <= full; S++) {
    int v = __builtin_ctz(S);
    unsigned int S2 = S & ~(1u << v);
    unsigned long long t = f[S2];
    unsigned int nb = adj[v] & S2;
    while (nb) {
      int u = __builtin_ctz(nb);
      t += f[S2 & ~(1u << u)];
      nb &= nb - 1;
    }
    f[S] = t;
  }
  return f[full];
}

int main(void) {
  char line[256];
  unsigned int adj[32];
  unsigned long long best = 0, count = 0;
  char bestg[256] = "";
  while (fgets(line, sizeof line, stdin)) {
    line[strcspn(line, "\r\n")] = 0;
    if (!line[0]) continue;
    int n = parse_graph6(line, adj);
    if (n <= 0) continue;
    unsigned long long z = hosoya(adj, n);
    count++;
    if (z > best) { best = z; strncpy(bestg, line, sizeof bestg - 1); }
  }
  printf("%llu %llu %s\n", count, best, bestg);
  return 0;
}
