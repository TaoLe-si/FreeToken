#include "hip_cmath_guard.h"
// ROCm/HIP iGPU MoE routing server (real HIP dispatch, minimal includes, 2026-08-30).

#include <hip/hip_runtime_api.h>
#include <windows.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>

#define HIP_CHECK(call) do { hipError_t err = call; if (err != hipSuccess) { fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)err, hipGetErrorString(err), __FILE__, __LINE__); return 1; } } while(0)

extern "C" hipError_t launch_moe_route(
    const float* d_routerW, const float* d_hidden,
    unsigned* d_top8_idx, float* d_top8_w,
    int E, int H, hipStream_t stream);

static int readN(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    size_t got = 0;
    while (got < n) {
        int r = _read(fd, p + got, (unsigned int)(n - got));
        if (r <= 0) return 0;
        got += (size_t)r;
    }
    return 1;
}
static int readLine(int fd, char* out, int max_len) {
    int n = 0;
    while (n < max_len - 1) {
        char c;
        int r = _read(fd, &c, 1);
        if (r <= 0) return 0;
        if (c == '\n') break;
        out[n++] = c;
    }
    out[n] = 0;
    return 1;
}
static void writeAll(int fd, const void* buf, size_t n) {
    _write(fd, buf, (unsigned int)n);
    fflush(NULL);
}

static double now_ms() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}



int main() {
    fprintf(stderr, "t_mtp_moe_route_hip_server starting...\n");
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    int dev = 0;
    hipDevice_t device;
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipDeviceGet(&device, dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    fprintf(stderr, "device %d: %s (CC %d.%d, %d CUs)\n",
            dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    int E = 256, H = 2048;
    int sticky_loaded = 0;
    float* d_routerW = NULL;
    float* d_hidden = NULL;
    unsigned* d_top8_idx = NULL;
    float* d_top8_w = NULL;
    HIP_CHECK(hipMalloc((void**)&d_hidden, H * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_top8_idx, 8 * sizeof(unsigned)));
    HIP_CHECK(hipMalloc((void**)&d_top8_w, 8 * sizeof(float)));

    fprintf(stderr, "t_mtp_moe_route_hip_server ready\n");

    while (1) {
        char line[256];
        if (!readLine(0, line, 256)) break;
        if (line[0] == 0) continue;
        if (strcmp(line, "QUIT") == 0) { fprintf(stderr, "QUIT\n"); break; }
        if (strncmp(line, "MOE_ROUTE_LOAD", 14) == 0) {
            int new_E = 256, new_H = 2048;
            sscanf(line, "MOE_ROUTE_LOAD %d %d", &new_E, &new_H);
            E = new_E; H = new_H;
            size_t bodySize = (size_t)E * H * sizeof(float);
            void* body = malloc(bodySize);
            if (!readN(0, body, bodySize)) { fprintf(stderr, "MOE_ROUTE_LOAD: read fail\n"); free(body); continue; }
            if (d_routerW) HIP_CHECK(hipFree(d_routerW));
            HIP_CHECK(hipMalloc((void**)&d_routerW, bodySize));
            HIP_CHECK(hipMemcpy(d_routerW, body, bodySize, hipMemcpyHostToDevice));
            sticky_loaded = 1;
            writeAll(1, "OK\n", 3);
            fprintf(stderr, "MOE_ROUTE_LOAD E=%d H=%d body=%zu bytes (HIP upload)\n", E, H, bodySize);
            free(body);
            continue;
        }
        if (strcmp(line, "MOE_ROUTE_FORWARD") == 0) {
            if (!sticky_loaded) { fprintf(stderr, "MOE_ROUTE_FORWARD before MOE_ROUTE_LOAD\n"); continue; }
            size_t hiddenSize = (size_t)H * sizeof(float);
            void* hiddenBytes = malloc(hiddenSize);
            if (!readN(0, hiddenBytes, hiddenSize)) { fprintf(stderr, "MOE_ROUTE_FORWARD: read fail\n"); free(hiddenBytes); continue; }
            HIP_CHECK(hipMemcpy(d_hidden, hiddenBytes, hiddenSize, hipMemcpyHostToDevice));
            double t0 = now_ms();
            hipError_t __kerr = launch_moe_route(d_routerW, d_hidden, d_top8_idx, d_top8_w, E, H, stream);
            if (__kerr != hipSuccess) { fprintf(stderr, "MOE_ROUTE_FORWARD launch: %s\n", hipGetErrorString(__kerr)); fflush(stderr); continue; }
            hipError_t __serr = hipStreamSynchronize(stream);
            if (__serr != hipSuccess) { fprintf(stderr, "MOE_ROUTE_FORWARD sync: %s\n", hipGetErrorString(__serr)); fflush(stderr); continue; }
            double t1 = now_ms();
            unsigned top8_idx[8];
            float top8_w[8];
            HIP_CHECK(hipMemcpy(top8_idx, d_top8_idx, 8 * sizeof(unsigned), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(top8_w, d_top8_w, 8 * sizeof(float), hipMemcpyDeviceToHost));
            writeAll(1, top8_idx, 8 * sizeof(unsigned));
            writeAll(1, top8_w, 8 * sizeof(float));
            fprintf(stderr, "MOE_ROUTE_FORWARD done (HIP dispatch, %.2f ms)\n", t1 - t0);
            free(hiddenBytes);
            continue;
        }
        fprintf(stderr, "unknown cmd: %s\n", line);
    }
    if (d_routerW) hipFree(d_routerW);
    hipFree(d_hidden);
    hipFree(d_top8_idx);
    hipFree(d_top8_w);
    hipStreamDestroy(stream);
    return 0;
}
