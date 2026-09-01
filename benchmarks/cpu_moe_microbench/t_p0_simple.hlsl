StructuredBuffer<uint>    packed : register(t0);
StructuredBuffer<float>   scl    : register(t1);
StructuredBuffer<float>   act    : register(t2);
StructuredBuffer<float>   bias   : register(t3);
RWStructuredBuffer<float> outv   : register(u0);
StructuredBuffer<float>   gbl    : register(t4);

cbuffer P : register(b0) { uint K; uint nbPerRow; uint nsPerRow; };

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          uint  t    : SV_GroupIndex,
          uint3 gr   : SV_GroupID) {
    if (t == 0) outv[gr.x] = bias[gr.x] + 1.0f;
}
