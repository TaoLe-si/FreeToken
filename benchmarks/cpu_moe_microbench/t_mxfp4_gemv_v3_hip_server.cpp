// HIP/ROCm iGPU MXFP4 GEMV server (port of t_mxfp4_gemv_v3_server.cpp, FC_LOAD/FC_CALL only).
//
// ROCm 6.4 / RDNA 3 (AMD Radeon 780M, gfx1103). Uses HIP API for device/queue/memcpy.
// Protocol: same as D3D12 v3 server, so existing IgpuFcStickyCPP can talk to it.
//
// 2026-08-30.

#include <hip/hip_runtime.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>

#define HIP_CHECK(call) do { hipError_t err = call; if (err != hipSuccess) { fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)err, hipGetErrorString(err), __FILE__, __LINE__); return 1; } } while(0)

extern "C" hipError_t launch_mxfp4_gemv_fc(
    const unsigned* d_packed, const float* d_scl, const float* d_bias_pb,
    const float* d_act, const float* d_gbl, const float* d_rowBias,
    float* d_outv, unsigned M, unsigned K, unsigned nbPerRow, unsigned nsPerRow,
    hipStream_t stream);

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
    fprintf(stderr, "t_mxfp4_gemv_v3_hip_server starting...\n");
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    fprintf(stderr, "device %d: %s (CC %d.%d, %d CUs)\n",
            dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    fprintf(stderr, "stream ok\n");

    // FC state
    unsigned M = 0, K = 0, nb = 0, ns = 0;
    int loaded = 0;
    float *d_packed = NULL, *d_scl = NULL, *d_bias_pb = NULL;
    float *d_act = NULL, *d_gbl = NULL, *d_rowBias = NULL, *d_outv = NULL;

    fprintf(stderr, "mxfp4-v3-hip server ready\n");

    while (1) {
        char line[256];
        if (!readLine(0, line, 256)) break;
        if (line[0] == 0) continue;
        if (strcmp(line, "QUIT") == 0) { fprintf(stderr, "QUIT\n"); break; }

        if (strncmp(line, "FC_LOAD", 7) == 0) {
            unsigned szP = 0, szS = 0, szB = 0;
            sscanf(line, "FC_LOAD %u %u %u %u %u", &M, &K, &szP, &szS, &szB);
            if ((K & 31u) != 0u) { fprintf(stderr, "FC_LOAD: K not mult of 32\n"); continue; }
            nb = K / 8u;
            ns = K / 32u;
            unsigned expectedP = M * nb * 4u, expectedS = M * ns * 4u, expectedB = M * ns * 4u;
            if (szP != expectedP || szS != expectedS || szB != expectedB) {
                fprintf(stderr, "FC_LOAD: size mismatch (got P=%u S=%u B=%u want %u/%u/%u)\n",
                        szP, szS, szB, expectedP, expectedS, expectedB);
                continue;
            }
            size_t bodySize = (size_t)(szP + szS + szB);
            void* body = malloc(bodySize);
            if (!readN(0, body, bodySize)) { fprintf(stderr, "FC_LOAD: body read fail\n"); free(body); continue; }

            // Release old
            if (d_packed) hipFree(d_packed);
            if (d_scl) hipFree(d_scl);
            if (d_bias_pb) hipFree(d_bias_pb);
            if (d_act) hipFree(d_act);
            if (d_gbl) hipFree(d_gbl);
            if (d_rowBias) hipFree(d_rowBias);
            if (d_outv) hipFree(d_outv);
            size_t szW = (size_t)M * nb * 4u, szSc = (size_t)M * ns * 4u, szBi = (size_t)M * ns * 4u;
            size_t szAct = (size_t)K * 4u, szGr = (size_t)M * 4u, szOut = (size_t)M * 4u;
            HIP_CHECK(hipMalloc((void**)&d_packed, szW));
            HIP_CHECK(hipMalloc((void**)&d_scl, szSc));
            HIP_CHECK(hipMalloc((void**)&d_bias_pb, szBi));
            HIP_CHECK(hipMalloc((void**)&d_act, szAct));
            HIP_CHECK(hipMalloc((void**)&d_gbl, szGr));
            HIP_CHECK(hipMalloc((void**)&d_rowBias, szGr));
            HIP_CHECK(hipMalloc((void**)&d_outv, szOut));

            // Upload body parts: packed + scl + bias_pb + gbl(all 1) + rowBias(all 0)
            HIP_CHECK(hipMemcpy(d_packed, body, szW, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_scl, (char*)body + szW, szSc, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_bias_pb, (char*)body + szW + szSc, szBi, hipMemcpyHostToDevice));
            // gbl = all 1, rowBias = all 0 (upload zeroed buffers, then overwrite gbl with 1)
            float* zeros = (float*)calloc(M, sizeof(float));
            HIP_CHECK(hipMemcpy(d_rowBias, zeros, szGr, hipMemcpyHostToDevice));
            free(zeros);
            float* ones = (float*)malloc(szGr);
            for (unsigned i = 0; i < M; i++) ones[i] = 1.0f;
            HIP_CHECK(hipMemcpy(d_gbl, ones, szGr, hipMemcpyHostToDevice));
            free(ones);

            loaded = 1;
            free(body);
            writeAll(1, "OK\n", 3);
            fprintf(stderr, "FC_LOAD M=%u K=%u done (%.1f MB)\n", M, K, (double)(szW + szSc + szBi + szGr * 2) / 1048576.0);
            continue;
        }

        if (strncmp(line, "FC_CALL", 7) == 0) {
            unsigned szA = 0;
            sscanf(line, "FC_CALL %u", &szA);
            if (!loaded) { fprintf(stderr, "FC_CALL: no FC loaded\n"); continue; }
            if ((size_t)szA != (size_t)K * 4u) { fprintf(stderr, "FC_CALL: szA=%u want %u\n", szA, K * 4u); continue; }
            void* act = malloc(szA);
            if (!readN(0, act, szA)) { fprintf(stderr, "FC_CALL: act read fail\n"); free(act); continue; }
            HIP_CHECK(hipMemcpy(d_act, act, szA, hipMemcpyHostToDevice));
            free(act);
            double t0 = now_ms();
            // nbPerRow = number of 32-elem blocks per row = K/32 = ns
            // nb = K/8 = number of uints per row (for packed size)
            hipError_t kerr = launch_mxfp4_gemv_fc(
                (const unsigned*)d_packed, (const float*)d_scl, (const float*)d_bias_pb,
                (const float*)d_act, (const float*)d_gbl, (const float*)d_rowBias,
                d_outv, M, K, ns, ns, stream);
            if (kerr != hipSuccess) { fprintf(stderr, "FC_CALL: launch failed: %s\n", hipGetErrorString(kerr)); continue; }
            hipError_t serr = hipStreamSynchronize(stream);
            if (serr != hipSuccess) { fprintf(stderr, "FC_CALL: sync failed: %s\n", hipGetErrorString(serr)); continue; }
            double t1 = now_ms();
            uint32_t szOut = M * 4u;
            float* h_outv = (float*)malloc(szOut);
            HIP_CHECK(hipMemcpy(h_outv, d_outv, szOut, hipMemcpyDeviceToHost));
            writeAll(1, &szOut, 4);
            writeAll(1, h_outv, szOut);
            free(h_outv);
            fprintf(stderr, "FC_CALL done (HIP dispatch, %.2f ms)\n", t1 - t0);
            continue;
        }

        fprintf(stderr, "unknown cmd: %s\n", line);
    }

    if (d_packed) hipFree(d_packed);
    if (d_scl) hipFree(d_scl);
    if (d_bias_pb) hipFree(d_bias_pb);
    if (d_act) hipFree(d_act);
    if (d_gbl) hipFree(d_gbl);
    if (d_rowBias) hipFree(d_rowBias);
    if (d_outv) hipFree(d_outv);
    hipStreamDestroy(stream);
    return 0;
}
