#include "solve.h"
#include "math.h"


inline float interp(const float* image, int h, int w, float x, float y) {
    x = x < 0 ? 0 : (x > w - 1.001 ? w - 1.001 : x);
    y = y < 0 ? 0 : (y > h - 1.001 ? h - 1.001 : y);
    int x0 = int(x), y0 = int(y);
    int x1 = x0 + 1, y1 = y0 + 1;
    float dx0 = x - x0, dy0 = y - y0;
    float dx1 = 1 - dx0, dy1 = 1 - dy0;
    float out = image[y0 * w + x0] * dx1 * dy1 +
                image[y0 * w + x1] * dx0 * dy1 +
                image[y1 * w + x0] * dx1 * dy0 +
                image[y1 * w + x1] * dx0 * dy0;
    return out;
}


void nms(float* out, const float* edge, const float* ori, int r, int s, float m, int w, int h) {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            float e = out[y * w + x] = edge[y * w + x];
            if (e == 0) {
                continue;
            }
            e *= m;
            float cos_o = cos(ori[y * w + x]);
            float sin_o = sin(ori[y * w + x]);
            for (int d = -r; d <= r; ++d) {
                if (d != 0) {
                    float e0 = interp(edge, h, w, x + d * cos_o, y + d * sin_o);
                    if (e < e0) {
                        out[y * w + x] = 0;
                        break;
                    }
                }
            }
        }
    }


    s = s > w / 2 ? w / 2 : s;
    s = s > h / 2 ? h / 2 : s;
    for (int x = 0; x < s; ++x) {
        for (int y = 0; y < h; ++y) {
            out[y * w + x] *= float(x) / s;
            out[y * w + w - 1 - x] *= float(x) / s;
        }
    }
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < s; ++y) {
            out[y * w + x] *= float(y) / s;
            out[(h - 1 - y) * w + x] *= float(y) / s;
        }
    }
}