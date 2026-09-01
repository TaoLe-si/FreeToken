StructuredBuffer<uint>    packed  : register(t0);
StructuredBuffer<float>   scl     : register(t1);
StructuredBuffer<float>   bias_pb : register(t2);
StructuredBuffer<float>   act     : register(t3);
StructuredBuffer<float>   gbl     : register(t4);
StructuredBuffer<float>   rowBias : register(t5);
RWStructuredBuffer<float> outv    : register(u0);
cbuffer P : register(b0) { uint K; uint nbPerRow; uint nsPerRow; };

// Test function that should NOT be optimized away
int getE2M1(uint n) {
    // Use switch which DXC may not optimize away
    switch(n) {
        case 0: return 0;
        case 1: return 1;
        case 2: return 2;
        case 3: return 3;
        case 4: return 4;
        case 5: return 6;
        case 6: return 8;
        case 7: return 12;
        case 8: return 0;
        case 9: return -1;
        case 10: return -2;
        case 11: return -3;
        case 12: return -4;
        case 13: return -6;
        case 14: return -8;
        case 15: return -12;
        default: return 0;
    }
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint t : SV_GroupIndex, uint3 gr : SV_GroupID) {
    uint row = gr.x;
    if (t == 0) {
        uint w = packed[row * nbPerRow];
        uint n0 = w & 0xFu;
        uint n1 = (w >> 4) & 0xFu;
        
        // Test 1: raw nibbles
        outv[0] = (float)n0;
        outv[1] = (float)n1;
        
        // Test 2: getE2M1 with switch
        outv[2] = (float)getE2M1(n0);
        outv[3] = (float)getE2M1(n1);
        
        // Test 3: inline if-else
        int w0;
        if (n0 == 0) w0 = 0;
        else if (n0 == 1) w0 = 1;
        else if (n0 == 2) w0 = 2;
        else if (n0 == 3) w0 = 3;
        else if (n0 == 4) w0 = 4;
        else if (n0 == 5) w0 = 6;
        else if (n0 == 6) w0 = 8;
        else if (n0 == 7) w0 = 12;
        else if (n0 == 8) w0 = 0;
        else if (n0 == 9) w0 = -1;
        else if (n0 == 10) w0 = -2;
        else if (n0 == 11) w0 = -3;
        else if (n0 == 12) w0 = -4;
        else if (n0 == 13) w0 = -6;
        else if (n0 == 14) w0 = -8;
        else w0 = -12;
        outv[4] = (float)w0;
        
        // Test 4: inline ternary
        outv[5] = (float)(n0 == 0 ? 0 :
                          n0 == 1 ? 1 :
                          n0 == 2 ? 2 :
                          n0 == 3 ? 3 :
                          n0 == 4 ? 4 :
                          n0 == 5 ? 6 :
                          n0 == 6 ? 8 :
                          n0 == 7 ? 12 :
                          n0 == 8 ? 0 :
                          n0 == 9 ? -1 :
                          n0 == 10 ? -2 :
                          n0 == 11 ? -3 :
                          n0 == 12 ? -4 :
                          n0 == 13 ? -6 :
                          n0 == 14 ? -8 : -12);
        
        // Test 5: act reads
        outv[6] = act[row * K + 0];
        outv[7] = act[row * K + 1];
    }
}
