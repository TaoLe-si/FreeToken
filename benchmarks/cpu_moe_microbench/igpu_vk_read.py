# iGPU SM read bandwidth via Vulkan compute shader (SPIR-V generated)
import vulkan as vk
import ctypes, time, sys

app = vk.VkApplicationInfo(sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName=b"bw", applicationVersion=1, pEngineName=b"bw", engineVersion=1,
    apiVersion=(1 << 22) | (2 << 12))
info = vk.VkInstanceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app)
inst = vk.vkCreateInstance(info, None)
phys = vk.vkEnumeratePhysicalDevices(inst)
import os
want = os.environ.get("VK_DEV", "0x1002")
want = int(want, 16)
target = None
for p in phys:
    props = vk.vkGetPhysicalDeviceProperties(p)
    if props.vendorID == want:
        target = p
if target is None: raise SystemExit("no device " + hex(want))
print("device:", str(vk.vkGetPhysicalDeviceProperties(target).deviceName).split(chr(0))[0])

qprops = vk.vkGetPhysicalDeviceQueueFamilyProperties(target)
qf = 0
for i, q in enumerate(qprops):
    if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT: qf = i; break
qci = vk.VkDeviceQueueCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    queueFamilyIndex=qf, queueCount=1, pQueuePriorities=[1.0])
dci = vk.VkDeviceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    queueCreateInfoCount=1, pQueueCreateInfos=[qci])
dev = vk.vkCreateDevice(target, dci, None)
queue = vk.vkGetDeviceQueue(dev, qf, 0)

memprops = vk.vkGetPhysicalDeviceMemoryProperties(target)
hv = None
for i, mt in enumerate(memprops.memoryTypes):
    if mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT and \
       mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:
        hv = i; break

def make_buf(size, usage):
    bci = vk.VkBufferCreateInfo(sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=size, usage=usage, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE)
    buf = vk.vkCreateBuffer(dev, bci, None)
    req = vk.vkGetBufferMemoryRequirements(dev, buf)
    aci = vk.VkMemoryAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=req.size, memoryTypeIndex=hv)
    mem = vk.vkAllocateMemory(dev, aci, None)
    vk.vkBindBufferMemory(dev, buf, mem, 0)
    return buf, mem

# shader module
spv = open("read_bw2.spv", "rb").read()
print("spv bytes:", len(spv))
spv_mv = memoryview(bytearray(spv))   # 必须保持引用存活
sci = vk.VkShaderModuleCreateInfo(sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    codeSize=len(spv), pCode=spv_mv)
print("sci created")
sm = vk.vkCreateShaderModule(dev, sci, None)
print("shader module ok")

# descriptor set layout: b1 = storage readonly (data), b2 = storage (dummy)
print("step: dsl")
b1 = vk.VkDescriptorSetLayoutBinding(binding=1, descriptorType=6, descriptorCount=1, stageFlags=32)
b2 = vk.VkDescriptorSetLayoutBinding(binding=2, descriptorType=6, descriptorCount=1, stageFlags=32)
dsl_info = vk.VkDescriptorSetLayoutCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    bindingCount=2, pBindings=[b1, b2])
dsl = vk.vkCreateDescriptorSetLayout(dev, dsl_info, None)
print("step: pl")
pl_info = vk.VkPipelineLayoutCreateInfo(sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    setLayoutCount=1, pSetLayouts=[dsl])
pl = vk.vkCreatePipelineLayout(dev, pl_info, None)
print("step: pipeline")
stage = vk.VkPipelineShaderStageCreateInfo(sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    stage=32, module=sm, pName="main")
cpi = vk.VkComputePipelineCreateInfo(sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    stage=stage, layout=pl)
pipe = vk.vkCreateComputePipelines(dev, None, 1, [cpi], None)[0]
print("pipeline ok")

# pool + descriptor set
psize = vk.VkDescriptorPoolSize(type=6, descriptorCount=2)
pool_info = vk.VkDescriptorPoolCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    maxSets=1, poolSizeCount=1, pPoolSizes=[psize])
pool = vk.vkCreateDescriptorPool(dev, pool_info, None)
print("step: ds alloc")
ds_info = vk.VkDescriptorSetAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    descriptorPool=pool, descriptorSetCount=1, pSetLayouts=[dsl])
ds = vk.vkAllocateDescriptorSets(dev, ds_info, None)[0]
print("step: ds alloc done")

# command pool/buffer
cpi2 = vk.VkCommandPoolCreateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex=qf)
pool2 = vk.vkCreateCommandPool(dev, cpi2, None)
abci = vk.VkCommandBufferAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    commandPool=pool2, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1)
cmdbuf = vk.vkAllocateCommandBuffers(dev, abci, None)[0]

def bench(size_gb, wgs, iters=6):
    size = size_gb << 30
    data, mdata = make_buf(size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    dum, mdum = make_buf(256 << 10, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    # 初始化 data（fill）
    bci0 = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
    vk.vkBeginCommandBuffer(cmdbuf, bci0)
    vk.vkCmdFillBuffer(cmdbuf, data, 0, size, 0x5A5A5A5A)
    vk.vkEndCommandBuffer(cmdbuf)
    si0 = vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=[cmdbuf])
    vk.vkQueueSubmit(queue, 1, [si0], None)
    vk.vkDeviceWaitIdle(dev)
    # 更新 descriptor
    dbi_data = vk.VkDescriptorBufferInfo(buffer=data, offset=0, range=size)
    dbi_dum = vk.VkDescriptorBufferInfo(buffer=dum, offset=0, range=256 << 10)
    w1 = vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, dstSet=ds,
        dstBinding=1, descriptorCount=1, descriptorType=6, pBufferInfo=[dbi_data])
    w2 = vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, dstSet=ds,
        dstBinding=2, descriptorCount=1, descriptorType=6, pBufferInfo=[dbi_dum])
    vk.vkUpdateDescriptorSets(dev, 2, [w1, w2], 0, None)
    best = float("inf")
    for _ in range(iters):
        vk.vkBeginCommandBuffer(cmdbuf, bci0)
        vk.vkCmdBindPipeline(cmdbuf, 5, pipe)
        vk.vkCmdBindDescriptorSets(cmdbuf, 5, pl, 0, 1, [ds], 0, None)
        vk.vkCmdDispatch(cmdbuf, wgs, 1, 1)
        vk.vkEndCommandBuffer(cmdbuf)
        si = vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=[cmdbuf])
        t0 = time.perf_counter()
        vk.vkQueueSubmit(queue, 1, [si], None)
        vk.vkDeviceWaitIdle(dev)
        best = min(best, time.perf_counter() - t0)
    # 每个 work-item 读 32 uint = 128B；总读 = wgs*256*128
    total = wgs * 256 * 128
    print(f"read {total/1e9:5.2f} GB (wgs {wgs}): {total/best/1e9:7.1f} GB/s")
    vk.vkFreeMemory(dev, mdata, None); vk.vkFreeMemory(dev, mdum, None)
    vk.vkDestroyBuffer(dev, data, None); vk.vkDestroyBuffer(dev, dum, None)

bench(2, 4096)    # 128MB
bench(2, 16384)   # 512MB
bench(2, 65536)   # 2GB
