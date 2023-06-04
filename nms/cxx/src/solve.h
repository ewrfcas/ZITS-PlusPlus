#include "csa.hh"

extern "C" {
    void solve(int n, int m, const int* graph, int* out_graph);
}

extern "C" {
    void nms(float* out, const float* edge, const float* ori, int r, int s, float m, int w, int h);
}

