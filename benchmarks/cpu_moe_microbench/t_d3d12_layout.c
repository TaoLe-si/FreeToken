#include <d3d12.h>
#include <stdio.h>
int main(void) {
    printf("RESOURCE_DESC sizeof=%zu offsetof(Flags)=%zu\n", sizeof(D3D12_RESOURCE_DESC), offsetof(D3D12_RESOURCE_DESC, Flags));
    printf("HEAP_PROPERTIES sizeof=%zu\n", sizeof(D3D12_HEAP_PROPERTIES));
    printf("COMPUTE_PSO sizeof=%zu\n", sizeof(D3D12_COMPUTE_PIPELINE_STATE_DESC));
    printf("ROOT_SIG_DESC sizeof=%zu\n", sizeof(D3D12_ROOT_SIGNATURE_DESC));
    printf("COMMAND_QUEUE_DESC sizeof=%zu\n", sizeof(D3D12_COMMAND_QUEUE_DESC));
    printf("RESOURCE_BARRIER sizeof=%zu transition sizeof=%zu\n", sizeof(D3D12_RESOURCE_BARRIER), sizeof(D3D12_RESOURCE_TRANSITION_BARRIER));
    printf("UAV_DESC sizeof=%zu buffer sizeof=%zu\n", sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC), sizeof(D3D12_BUFFER_UAV));
    printf("SHADER_BYTECODE sizeof=%zu\n", sizeof(D3D12_SHADER_BYTECODE));
    printf("STATE: UAV=0x%x COPY_SOURCE=0x%x COPY_DEST=0x%x\n", D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    return 0;
}
