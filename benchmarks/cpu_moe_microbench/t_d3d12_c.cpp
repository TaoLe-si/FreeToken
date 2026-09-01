#include <d3d12.h>
// d3dx12.h not found; manual desc
static D3D12_RESOURCE_DESC make_buffer_desc(UINT64 size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.SampleDesc.Quality = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <stdio.h>
#include <string.h>
using namespace Microsoft::WRL;
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

int main() {
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) { printf("factory fail %08X\n", hr); return 1; }
    ComPtr<IDXGIAdapter1> adapter;
    hr = factory->EnumAdapters1(0, &adapter);
    if (FAILED(hr)) { printf("enum fail %08X\n", hr); return 1; }
    DXGI_ADAPTER_DESC1 d;
    adapter->GetDesc1(&d);
    printf("adapter: %ls vid=%04X\n", d.Description, d.VendorId);
    ComPtr<ID3D12Device> dev;
    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
    if (FAILED(hr)) { printf("dev fail %08X\n", hr); return 1; }
    printf("device ok\n");
    ComPtr<ID3D12Resource> uav;
    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                 D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    D3D12_RESOURCE_DESC rd = make_buffer_desc(1 << 20, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                      IID_PPV_ARGS(&uav));
    printf("CreateCommittedResource: %08X %s\n", hr, FAILED(hr) ? "FAIL" : "OK");
    ComPtr<ID3D12Resource> rb;
    D3D12_RESOURCE_DESC rd2 = make_buffer_desc(1 << 20, D3D12_RESOURCE_FLAG_NONE);
    hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd2,
                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                      IID_PPV_ARGS(&rb));
    printf("readback resource: %08X %s\n", hr, FAILED(hr) ? "FAIL" : "OK");
    return 0;
}
