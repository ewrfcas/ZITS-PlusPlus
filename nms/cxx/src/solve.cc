#include "solve.h"


void solve(int n, int m, const int* graph, int* out_graph) {
    CSA csa(2 * n, m, graph);
    assert(csa.edges() == n);
    for (int i = 0; i < n; ++i) {
        int a, b, c;
        csa.edge(i, a, b, c);
        out_graph[i * 3 + 0] = a - 1;
        out_graph[i * 3 + 1] = b - 1 - n;
        out_graph[i * 3 + 2] = c;
    }
}

