// MXFP4 GEMV (FC broadcast) - NVFP4, act shared across all rows (act[ai], no row stride).
// For sticky FC: M output rows x K inputs, one shared activation vector of K floats.
StructuredBuffer<uint>    packed  : register(t0);
StructuredBuffer<float>   scl     : register(t1);
StructuredBuffer<float>   bias_pb : register(t2);
StructuredBuffer<float>   act     : register(t3);
StructuredBuffer<float>   gbl     : register(t4);
StructuredBuffer<float>   rowBias : register(t5);
RWStructuredBuffer<float> outv    : register(u0);
cbuffer P : register(b0) { uint K; uint nbPerRow; uint nsPerRow; };

groupshared float sh[256];

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          uint  t    : SV_GroupIndex,
          uint3 gr   : SV_GroupID) {
    uint row = gr.x;
    float acc = 0.0f;

    for (uint b = t; b < nbPerRow; b += 256) {
        uint pbase = row * nbPerRow + b;
        uint sbase = row * nsPerRow + b;
        float bs2 = scl[sbase];
        float bb = bias_pb[sbase];
        
        float wsum = 0.0f;
        uint ai_base = b * 32u;
        
        // Process 4 uints = 32 nibbles
        for (int wordIdx = 0; wordIdx < 4; wordIdx++) {
            uint w = packed[pbase * 4u + (uint)wordIdx];
            uint ai = ai_base + (uint)wordIdx * 8u;
            uint n0 = w & 0xFu; uint n1 = (w >> 4) & 0xFu;
            uint n2 = (w >> 8) & 0xFu; uint n3 = (w >> 12) & 0xFu;
            uint n4 = (w >> 16) & 0xFu; uint n5 = (w >> 20) & 0xFu;
            uint n6 = (w >> 24) & 0xFu; uint n7 = (w >> 28) & 0xFu;
            
            float a0 = act[ai + 0u];
            float a1 = act[ai + 1u];
            float a2 = act[ai + 2u];
            float a3 = act[ai + 3u];
            float a4 = act[ai + 4u];
            float a5 = act[ai + 5u];
            float a6 = act[ai + 6u];
            float a7 = act[ai + 7u];
            
            // Ternary chain per nibble
            wsum += (n0==0u?0.0f:n0==1u?1.0f:n0==2u?2.0f:n0==3u?3.0f:n0==4u?4.0f:n0==5u?6.0f:n0==6u?8.0f:n0==7u?12.0f:n0==8u?0.0f:n0==9u?-1.0f:n0==10u?-2.0f:n0==11u?-3.0f:n0==12u?-4.0f:n0==13u?-6.0f:n0==14u?-8.0f:-12.0f)*a0;
            wsum += (n1==0u?0.0f:n1==1u?1.0f:n1==2u?2.0f:n1==3u?3.0f:n1==4u?4.0f:n1==5u?6.0f:n1==6u?8.0f:n1==7u?12.0f:n1==8u?0.0f:n1==9u?-1.0f:n1==10u?-2.0f:n1==11u?-3.0f:n1==12u?-4.0f:n1==13u?-6.0f:n1==14u?-8.0f:-12.0f)*a1;
            wsum += (n2==0u?0.0f:n2==1u?1.0f:n2==2u?2.0f:n2==3u?3.0f:n2==4u?4.0f:n2==5u?6.0f:n2==6u?8.0f:n2==7u?12.0f:n2==8u?0.0f:n2==9u?-1.0f:n2==10u?-2.0f:n2==11u?-3.0f:n2==12u?-4.0f:n2==13u?-6.0f:n2==14u?-8.0f:-12.0f)*a2;
            wsum += (n3==0u?0.0f:n3==1u?1.0f:n3==2u?2.0f:n3==3u?3.0f:n3==4u?4.0f:n3==5u?6.0f:n3==6u?8.0f:n3==7u?12.0f:n3==8u?0.0f:n3==9u?-1.0f:n3==10u?-2.0f:n3==11u?-3.0f:n3==12u?-4.0f:n3==13u?-6.0f:n3==14u?-8.0f:-12.0f)*a3;
            wsum += (n4==0u?0.0f:n4==1u?1.0f:n4==2u?2.0f:n4==3u?3.0f:n4==4u?4.0f:n4==5u?6.0f:n4==6u?8.0f:n4==7u?12.0f:n4==8u?0.0f:n4==9u?-1.0f:n4==10u?-2.0f:n4==11u?-3.0f:n4==12u?-4.0f:n4==13u?-6.0f:n4==14u?-8.0f:-12.0f)*a4;
            wsum += (n5==0u?0.0f:n5==1u?1.0f:n5==2u?2.0f:n5==3u?3.0f:n5==4u?4.0f:n5==5u?6.0f:n5==6u?8.0f:n5==7u?12.0f:n5==8u?0.0f:n5==9u?-1.0f:n5==10u?-2.0f:n5==11u?-3.0f:n5==12u?-4.0f:n5==13u?-6.0f:n5==14u?-8.0f:-12.0f)*a5;
            wsum += (n6==0u?0.0f:n6==1u?1.0f:n6==2u?2.0f:n6==3u?3.0f:n6==4u?4.0f:n6==5u?6.0f:n6==6u?8.0f:n6==7u?12.0f:n6==8u?0.0f:n6==9u?-1.0f:n6==10u?-2.0f:n6==11u?-3.0f:n6==12u?-4.0f:n6==13u?-6.0f:n6==14u?-8.0f:-12.0f)*a6;
            wsum += (n7==0u?0.0f:n7==1u?1.0f:n7==2u?2.0f:n7==3u?3.0f:n7==4u?4.0f:n7==5u?6.0f:n7==6u?8.0f:n7==7u?12.0f:n7==8u?0.0f:n7==9u?-1.0f:n7==10u?-2.0f:n7==11u?-3.0f:n7==12u?-4.0f:n7==13u?-6.0f:n7==14u?-8.0f:-12.0f)*a7;
        }
        acc += (wsum + bb) * bs2;
    }

    sh[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) outv[row] = sh[0] * gbl[row] + rowBias[row];
}
