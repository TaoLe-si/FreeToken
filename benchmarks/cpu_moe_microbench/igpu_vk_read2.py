# iGPU SM read bandwidth via Vulkan compute (pure-cffi pipeline path)
import vulkan as vk
import vulkan._vulkan as vv
import time, struct, os
ffi = vv.ffi
lib = vv.lib

want = int(os.environ.get("VK_DEV", "0x1002"), 16)
app = vk.VkApplicationInfo(sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName=b"bw", applicationVersion=1, pEngineName=b"bw", engineVersion=1,
    apiVersion=(1 << 22) | (2 << 12))
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app), None)
phys = vk.vkEnumeratePhysicalDevices(inst)
target = next(p for p in phys if vk.vkGetPhysicalDeviceProperties(p).vendorID == want)
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
print("hv mem type:", hv)

print("step: building shader words")
# shader module（纯 cffi：uint32 数组拷贝）
spv = open("read_bw_atomic.spv", "rb").read()  # atomic 防消除
spv_words = list(struct.unpack("<%dI" % (len(spv) // 4), spv))
words_c = ffi.new("uint32_t[]", spv_words)
sci_c = ffi.new("VkShaderModuleCreateInfo*")
sci_c.sType = vv.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
sci_c.flags = 0; sci_c.codeSize = len(spv)
sci_c.pCode = words_c
print("step: calling vkCreateShaderModule")
sm = ffi.new("VkShaderModule*")
r = lib.vkCreateShaderModule(dev, sci_c, ffi.NULL, sm)
print("shader module:", r, sm[0])
assert r == 0

# ===== 纯 cffi：descriptor layout =====
bindings = ffi.new("VkDescriptorSetLayoutBinding[2]")
bindings[0].binding = 1; bindings[0].descriptorType = 6; bindings[0].descriptorCount = 1
bindings[0].stageFlags = 32; bindings[0].pImmutableSamplers = ffi.NULL
bindings[1].binding = 2; bindings[1].descriptorType = 6; bindings[1].descriptorCount = 1
bindings[1].stageFlags = 32; bindings[1].pImmutableSamplers = ffi.NULL
dsl_info = ffi.new("VkDescriptorSetLayoutCreateInfo*")
dsl_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
dsl_info.flags = 0; dsl_info.bindingCount = 2; dsl_info.pBindings = bindings
dsl = ffi.new("VkDescriptorSetLayout*")
r = lib.vkCreateDescriptorSetLayout(dev, dsl_info, ffi.NULL, dsl)
print("dsl:", r, dsl[0])
assert r == 0

# pipeline layout
pl_info = ffi.new("VkPipelineLayoutCreateInfo*")
pl_info.sType = vv.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
pl_info.flags = 0; pl_info.setLayoutCount = 1; pl_info.pSetLayouts = dsl
pl = ffi.new("VkPipelineLayout*")
r = lib.vkCreatePipelineLayout(dev, pl_info, ffi.NULL, pl)
print("pl:", r)
assert r == 0

# pipeline
print("step: build cpi")
name = ffi.new("char[]", b"main")
cpi_c = ffi.new("VkComputePipelineCreateInfo*")
cpi_c.sType = vv.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
cpi_c.pNext = ffi.NULL; cpi_c.flags = 0; cpi_c.layout = pl[0]
cpi_c.basePipelineHandle = ffi.NULL; cpi_c.basePipelineIndex = -1
cpi_c.stage.sType = vv.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
cpi_c.stage.pNext = ffi.NULL; cpi_c.stage.flags = 0
cpi_c.stage.stage = 32; cpi_c.stage.module = sm[0]
cpi_c.stage.pName = name; cpi_c.stage.pSpecializationInfo = ffi.NULL
print("step: call vkCreateComputePipelines")
pipe = ffi.new("VkPipeline*")
r = lib.vkCreateComputePipelines(dev, ffi.NULL, 1, cpi_c, ffi.NULL, pipe)
print("pipeline:", r)
assert r == 0

# descriptor pool
psize = ffi.new("VkDescriptorPoolSize[1]")
psize[0].type = 6; psize[0].descriptorCount = 2
pool_info = ffi.new("VkDescriptorPoolCreateInfo*")
pool_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
pool_info.flags = 0; pool_info.maxSets = 1; pool_info.poolSizeCount = 1
pool_info.pPoolSizes = psize
pool = ffi.new("VkDescriptorPool*")
r = lib.vkCreateDescriptorPool(dev, pool_info, ffi.NULL, pool)
print("pool:", r)
assert r == 0
ds_info = ffi.new("VkDescriptorSetAllocateInfo*")
ds_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
ds_info.descriptorPool = pool[0]; ds_info.descriptorSetCount = 1; ds_info.pSetLayouts = dsl
ds = ffi.new("VkDescriptorSet*")
r = lib.vkAllocateDescriptorSets(dev, ds_info, ds)
print("ds:", r)
assert r == 0

# command pool/buffer
cp_info = ffi.new("VkCommandPoolCreateInfo*")
cp_info.sType = vv.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
cp_info.flags = vv.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cp_info.queueFamilyIndex = qf
pool2 = ffi.new("VkCommandPool*")
r = lib.vkCreateCommandPool(dev, cp_info, ffi.NULL, pool2)
print("cmdpool:", r)
assert r == 0
ab_info = ffi.new("VkCommandBufferAllocateInfo*")
ab_info.sType = vv.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
ab_info.commandPool = pool2[0]; ab_info.level = 0; ab_info.commandBufferCount = 1
cmdbuf = ffi.new("VkCommandBuffer*")
r = lib.vkAllocateCommandBuffers(dev, ab_info, cmdbuf)
print("cmdbuf:", r)
assert r == 0
begin_info = ffi.new("VkCommandBufferBeginInfo*")
begin_info.sType = vv.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
begin_info.flags = 0; begin_info.pInheritanceInfo = ffi.NULL

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

def bench(size_gb, wgs, iters=5):
    size = size_gb << 30
    data, mdata = make_buf(size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    dum, mdum = make_buf(256 << 10, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    # fill data (GPU)
    r = lib.vkBeginCommandBuffer(cmdbuf[0], begin_info); assert r == 0
    lib.vkCmdFillBuffer(cmdbuf[0], data, 0, size, 0x5A5A5A5A)
    lib.vkCmdFillBuffer(cmdbuf[0], dum, 0, 256 << 10, 0xDEADBEEF)
    r = lib.vkEndCommandBuffer(cmdbuf[0]); assert r == 0
    si0 = ffi.new("VkSubmitInfo*")
    si0.sType = vv.VK_STRUCTURE_TYPE_SUBMIT_INFO; si0.commandBufferCount = 1
    si0.pCommandBuffers = cmdbuf
    lib.vkQueueSubmit(queue, 1, si0, ffi.NULL)
    lib.vkDeviceWaitIdle(dev)
    # update descriptors
    dbi1 = ffi.new("VkDescriptorBufferInfo[1]")
    dbi1[0].buffer = data; dbi1[0].offset = 0; dbi1[0].range = size
    dbi2 = ffi.new("VkDescriptorBufferInfo[1]")
    dbi2[0].buffer = dum; dbi2[0].offset = 0; dbi2[0].range = 256 << 10
    writes = ffi.new("VkWriteDescriptorSet[2]")
    writes[0].sType = vv.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
    writes[0].dstSet = ds[0]; writes[0].dstBinding = 1; writes[0].dstArrayElement = 0
    writes[0].descriptorCount = 1; writes[0].descriptorType = 6
    writes[0].pBufferInfo = dbi1
    writes[1].sType = vv.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
    writes[1].dstSet = ds[0]; writes[1].dstBinding = 2; writes[1].dstArrayElement = 0
    writes[1].descriptorCount = 1; writes[1].descriptorType = 6
    writes[1].pBufferInfo = dbi2
    lib.vkUpdateDescriptorSets(dev, 2, writes, 0, ffi.NULL)
    # CPU 校验：先 dispatch 一次（128 wgs），读回 out_data 若干值
    r = lib.vkBeginCommandBuffer(cmdbuf[0], begin_info); assert r == 0
    lib.vkCmdBindPipeline(cmdbuf[0], 5, pipe[0])
    lib.vkCmdBindDescriptorSets(cmdbuf[0], 5, pl[0], 0, 1, ds, 0, ffi.NULL)
    lib.vkCmdDispatch(cmdbuf[0], 128, 1, 1)
    r = lib.vkEndCommandBuffer(cmdbuf[0]); assert r == 0
    si_c = ffi.new("VkSubmitInfo*")
    si_c.sType = vv.VK_STRUCTURE_TYPE_SUBMIT_INFO; si_c.commandBufferCount = 1
    si_c.pCommandBuffers = cmdbuf
    lib.vkQueueSubmit(queue, 1, si_c, ffi.NULL)
    lib.vkDeviceWaitIdle(dev)
    # 校验：读回 data 前 4 值（确认 fill） + out 前 4 值
    for lbl, mem in [("data", mdata), ("out", mdum)]:
        ptr_out = ffi.new("void**")
        rmap = lib.vkMapMemory(dev, mem, 0, 4096, 0, ptr_out)
        if rmap == 0 and ptr_out[0] != ffi.NULL:
            vals = [ffi.cast("uint32_t*", ptr_out[0])[i] for i in range(4)]
            print(f"  {lbl}[0..3] =", [hex(v) for v in vals])
        lib.vkUnmapMemory(dev, mem)
    best = float("inf")
    for _ in range(iters):
        r = lib.vkBeginCommandBuffer(cmdbuf[0], begin_info); assert r == 0
        lib.vkCmdBindPipeline(cmdbuf[0], 5, pipe[0])
        lib.vkCmdBindDescriptorSets(cmdbuf[0], 5, pl[0], 0, 1, ds, 0, ffi.NULL)
        lib.vkCmdDispatch(cmdbuf[0], wgs, 1, 1)
        r = lib.vkEndCommandBuffer(cmdbuf[0]); assert r == 0
        si = ffi.new("VkSubmitInfo*")
        si.sType = vv.VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1
        si.pCommandBuffers = cmdbuf
        t0 = time.perf_counter()
        lib.vkQueueSubmit(queue, 1, si, ffi.NULL)
        lib.vkDeviceWaitIdle(dev)
        best = min(best, time.perf_counter() - t0)
    total = wgs * 256 * 4096
    print(f"read {total/1e9:5.2f} GB (wgs {wgs}): {total/best/1e9:7.1f} GB/s")
    vk.vkFreeMemory(dev, mdata, None); vk.vkFreeMemory(dev, mdum, None)
    vk.vkDestroyBuffer(dev, data, None); vk.vkDestroyBuffer(dev, dum, None)

# v6: 每项 4KB
bench(2, 512)      # 512MB
bench(2, 1024)     # 1GB
bench(2, 2048)     # 2GB
# 注意 262144 wgs：262144*256*8 = 536M uint = 2.1GB > 2GB buffer → 越界！用 131072（1.07GB）
