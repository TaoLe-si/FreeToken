# B 组 iGPU GEMV（W4A8 NVFP4）原型：正确性（CPU 参考）+ 带宽
# 用法: python igpu_gemv.py [M] [K] [iters]
import vulkan as vk
import vulkan._vulkan as vv
import time, struct, os, sys, random
ffi = vv.ffi
lib = vv.lib

M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NB = K // 16

want = int(os.environ.get("VK_DEV", "0x1002"), 16)
app = vk.VkApplicationInfo(sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName=b"gemv", applicationVersion=1, pEngineName=b"gemv", engineVersion=1,
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
dev = vk.vkCreateDevice(target, vk.VkDeviceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    queueCreateInfoCount=1, pQueueCreateInfos=[qci]), None)
queue = vk.vkGetDeviceQueue(dev, qf, 0)
memprops = vk.vkGetPhysicalDeviceMemoryProperties(target)
hv = None
for i, mt in enumerate(memprops.memoryTypes):
    if mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT and \
       mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:
        hv = i; break

spv = open(os.path.join(os.path.dirname(__file__), "igpu_gemv.spv"), "rb").read()
spv_words = list(struct.unpack("<%dI" % (len(spv) // 4), spv))
words_c = ffi.new("uint32_t[]", spv_words)
sci_c = ffi.new("VkShaderModuleCreateInfo*")
sci_c.sType = vv.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
sci_c.flags = 0; sci_c.codeSize = len(spv); sci_c.pCode = words_c
sm = ffi.new("VkShaderModule*")
assert lib.vkCreateShaderModule(dev, sci_c, ffi.NULL, sm) == 0

# 5 storage bindings (0-4) + push constant
bindings = ffi.new("VkDescriptorSetLayoutBinding[5]")
for bi in range(5):
    bindings[bi].binding = bi; bindings[bi].descriptorType = 6
    bindings[bi].descriptorCount = 1; bindings[bi].stageFlags = 32
    bindings[bi].pImmutableSamplers = ffi.NULL
dsl_info = ffi.new("VkDescriptorSetLayoutCreateInfo*")
dsl_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
dsl_info.flags = 0; dsl_info.bindingCount = 5; dsl_info.pBindings = bindings
dsl = ffi.new("VkDescriptorSetLayout*")
assert lib.vkCreateDescriptorSetLayout(dev, dsl_info, ffi.NULL, dsl) == 0

pc_range = ffi.new("VkPushConstantRange[1]")
pc_range[0].stageFlags = 32; pc_range[0].offset = 0; pc_range[0].size = 12  # 2*uint + float
pl_info = ffi.new("VkPipelineLayoutCreateInfo*")
pl_info.sType = vv.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
pl_info.flags = 0; pl_info.setLayoutCount = 1; pl_info.pSetLayouts = dsl
pl_info.pushConstantRangeCount = 1; pl_info.pPushConstantRanges = pc_range
pl = ffi.new("VkPipelineLayout*")
assert lib.vkCreatePipelineLayout(dev, pl_info, ffi.NULL, pl) == 0

name = ffi.new("char[]", b"main")
cpi_c = ffi.new("VkComputePipelineCreateInfo*")
cpi_c.sType = vv.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
cpi_c.pNext = ffi.NULL; cpi_c.flags = 0; cpi_c.layout = pl[0]
cpi_c.basePipelineHandle = ffi.NULL; cpi_c.basePipelineIndex = -1
cpi_c.stage.sType = vv.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
cpi_c.stage.pNext = ffi.NULL; cpi_c.stage.flags = 0
cpi_c.stage.stage = 32; cpi_c.stage.module = sm[0]
cpi_c.stage.pName = name; cpi_c.stage.pSpecializationInfo = ffi.NULL
pipe = ffi.new("VkPipeline*")
assert lib.vkCreateComputePipelines(dev, ffi.NULL, 1, cpi_c, ffi.NULL, pipe) == 0

# ---- consumer pipeline（保留但未使用；条件写方案已替代）----
spv2 = open(os.path.join(os.path.dirname(__file__), "igpu_consume.spv"), "rb").read()
spv2_words = list(struct.unpack("<%dI" % (len(spv2) // 4), spv2))
words2_c = ffi.new("uint32_t[]", spv2_words)
sci2_c = ffi.new("VkShaderModuleCreateInfo*")
sci2_c.sType = vv.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
sci2_c.flags = 0; sci2_c.codeSize = len(spv2); sci2_c.pCode = words2_c
sm2 = ffi.new("VkShaderModule*")
assert lib.vkCreateShaderModule(dev, sci2_c, ffi.NULL, sm2) == 0
bindings2 = ffi.new("VkDescriptorSetLayoutBinding[2]")
for bi in range(2):
    bindings2[bi].binding = bi; bindings2[bi].descriptorType = 6
    bindings2[bi].descriptorCount = 1; bindings2[bi].stageFlags = 32
    bindings2[bi].pImmutableSamplers = ffi.NULL
dsl2_info = ffi.new("VkDescriptorSetLayoutCreateInfo*")
dsl2_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
dsl2_info.flags = 0; dsl2_info.bindingCount = 2; dsl2_info.pBindings = bindings2
dsl2 = ffi.new("VkDescriptorSetLayout*")
assert lib.vkCreateDescriptorSetLayout(dev, dsl2_info, ffi.NULL, dsl2) == 0
pc2_range = ffi.new("VkPushConstantRange[1]")
pc2_range[0].stageFlags = 32; pc2_range[0].offset = 0; pc2_range[0].size = 4
pl2_info = ffi.new("VkPipelineLayoutCreateInfo*")
pl2_info.sType = vv.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
pl2_info.flags = 0; pl2_info.setLayoutCount = 1; pl2_info.pSetLayouts = dsl2
pl2_info.pushConstantRangeCount = 1; pl2_info.pPushConstantRanges = pc2_range
pl2 = ffi.new("VkPipelineLayout*")
assert lib.vkCreatePipelineLayout(dev, pl2_info, ffi.NULL, pl2) == 0
cpi2_c = ffi.new("VkComputePipelineCreateInfo*")
cpi2_c.sType = vv.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
cpi2_c.pNext = ffi.NULL; cpi2_c.flags = 0; cpi2_c.layout = pl2[0]
cpi2_c.basePipelineHandle = ffi.NULL; cpi2_c.basePipelineIndex = -1
cpi2_c.stage.sType = vv.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
cpi2_c.stage.pNext = ffi.NULL; cpi2_c.stage.flags = 0
cpi2_c.stage.stage = 32; cpi2_c.stage.module = sm2[0]
cpi2_c.stage.pName = name; cpi2_c.stage.pSpecializationInfo = ffi.NULL
pipe2 = ffi.new("VkPipeline*")
assert lib.vkCreateComputePipelines(dev, ffi.NULL, 1, cpi2_c, ffi.NULL, pipe2) == 0
pool_info2 = ffi.new("VkDescriptorPoolCreateInfo*")
psize2 = ffi.new("VkDescriptorPoolSize[1]")
psize2[0].type = 6; psize2[0].descriptorCount = 2
pool_info2.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
pool_info2.flags = 0; pool_info2.maxSets = 1; pool_info2.poolSizeCount = 1
pool_info2.pPoolSizes = psize2
pool2d = ffi.new("VkDescriptorPool*")
assert lib.vkCreateDescriptorPool(dev, pool_info2, ffi.NULL, pool2d) == 0
ds2_info = ffi.new("VkDescriptorSetAllocateInfo*")
ds2_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
ds2_info.descriptorPool = pool2d[0]; ds2_info.descriptorSetCount = 1; ds2_info.pSetLayouts = dsl2
ds2 = ffi.new("VkDescriptorSet*")
assert lib.vkAllocateDescriptorSets(dev, ds2_info, ds2) == 0

psize = ffi.new("VkDescriptorPoolSize[1]")
psize[0].type = 6; psize[0].descriptorCount = 5
pool_info = ffi.new("VkDescriptorPoolCreateInfo*")
pool_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
pool_info.flags = 0; pool_info.maxSets = 1; pool_info.poolSizeCount = 1
pool_info.pPoolSizes = psize
pool = ffi.new("VkDescriptorPool*")
assert lib.vkCreateDescriptorPool(dev, pool_info, ffi.NULL, pool) == 0
ds_info = ffi.new("VkDescriptorSetAllocateInfo*")
ds_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
ds_info.descriptorPool = pool[0]; ds_info.descriptorSetCount = 1; ds_info.pSetLayouts = dsl
ds = ffi.new("VkDescriptorSet*")
assert lib.vkAllocateDescriptorSets(dev, ds_info, ds) == 0

cp_info = ffi.new("VkCommandPoolCreateInfo*")
cp_info.sType = vv.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
cp_info.flags = vv.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cp_info.queueFamilyIndex = qf
pool2 = ffi.new("VkCommandPool*")
assert lib.vkCreateCommandPool(dev, cp_info, ffi.NULL, pool2) == 0
ab_info = ffi.new("VkCommandBufferAllocateInfo*")
ab_info.sType = vv.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
ab_info.commandPool = pool2[0]; ab_info.level = 0; ab_info.commandBufferCount = 1
cmdbuf = ffi.new("VkCommandBuffer*")
assert lib.vkAllocateCommandBuffers(dev, ab_info, cmdbuf) == 0
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

def map_write(mem, data_bytes):
    ptr_out = ffi.new("void**")
    assert lib.vkMapMemory(dev, mem, 0, len(data_bytes), 0, ptr_out) == 0
    ffi.memmove(ptr_out[0], data_bytes, len(data_bytes))
    lib.vkUnmapMemory(dev, mem)

def map_read(mem, nbytes):
    ptr_out = ffi.new("void**")
    assert lib.vkMapMemory(dev, mem, 0, nbytes, 0, ptr_out) == 0
    b = ffi.buffer(ptr_out[0], nbytes)[:]
    lib.vkUnmapMemory(dev, mem)
    return b

# ---------- 数据（与 CPU 微基准同种子逻辑；act -127..127 排除 -128）----------
random.seed(42)
K_E2M1X2 = [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12]
E4M3 = {}
for i in range(256):
    s = -1.0 if (i & 0x80) else 1.0
    e = (i >> 3) & 0xF; m = i & 7
    E4M3[i] = s * (m / 8.0 * 2.0**-6 if e == 0 else (1.0 + m/8.0) * 2.0**(e-7))
def e4m3(i):
    s = -1.0 if (i & 0x80) else 1.0
    e = (i >> 3) & 0xF; m = i & 7
    return s * (m * 0.001953125 if e == 0 else (1.0 + m/8.0) * 2.0**(e-7))

packed = [random.randrange(256) for _ in range(M * NB * 8)]       # uint8
scale  = [random.randrange(256) for _ in range(M * NB)]           # e4m3 uint8
act8   = [random.randrange(255) - 127 for _ in range(K)]          # int8 -127..127
asb    = [0.01 + 0.05 * random.randrange(100)/100.0 for _ in range(NB)]
global_s = 0.25

# act 打包：每块 4 uint（[e0..3][e4..7][o0..3][o4..7] int8）
act_pack = []
for b in range(NB):
    ev = act8[b*16:b*16+8]; od = act8[b*16+8:b*16+16]
    for grp in (ev[0:4], ev[4:8], od[0:4], od[4:8]):
        u = 0
        for j in range(4):
            u |= (grp[j] & 0xFF) << (j*8)
        act_pack.append(u)

# CPU 参考 GEMV
def cpu_gemv():
    out = []
    for row in range(M):
        acc = 0.0
        wbase = row * NB * 8
        sbase = row * NB
        for b in range(NB):
            isum = 0
            for j in range(8):
                pk = packed[wbase + b*8 + j]
                isum += K_E2M1X2[pk & 0xF] * act8[b*16+j] + K_E2M1X2[pk >> 4] * act8[b*16+8+j]
            acc += e4m3(scale[sbase+b]) * asb[b] * isum
        out.append(acc * global_s * 0.5)
    return out

ref = cpu_gemv()

# ---------- GPU buffers ----------
USAGE = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
wbuf, wmem = make_buf(M*NB*8, USAGE)
sbuf, smem = make_buf(M*NB, USAGE)
abuf, amem = make_buf(NB*4*4, USAGE)
asbuf, asmem = make_buf(NB*4, USAGE)
obuf, omem = make_buf(M*4, USAGE)
map_write(omem, struct.pack("<%dI" % M, *([0x5A5A5A5A] * M)))  # 预填充检测 DCE
o2buf, o2mem = make_buf(M*4, USAGE)
map_write(o2mem, struct.pack("<%dI" % M, *([0] * M)))
map_write(wmem, bytes(packed))
map_write(smem, bytes(scale))
map_write(amem, struct.pack("<%dI" % len(act_pack), *act_pack))
map_write(asmem, struct.pack("<%df" % NB, *asb))

dbi = [ffi.new("VkDescriptorBufferInfo[1]") for _ in range(5)]
bufs = [(wbuf, M*NB*8), (sbuf, M*NB), (abuf, NB*4*4), (asbuf, NB*4), (obuf, M*4)]
for i, (b, sz) in enumerate(bufs):
    dbi[i][0].buffer = b; dbi[i][0].offset = 0; dbi[i][0].range = sz
writes = ffi.new("VkWriteDescriptorSet[5]")
for i in range(5):
    writes[i].sType = vv.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
    writes[i].dstSet = ds[0]; writes[i].dstBinding = i; writes[i].dstArrayElement = 0
    writes[i].descriptorCount = 1; writes[i].descriptorType = 6
    writes[i].pBufferInfo = dbi[i]
lib.vkUpdateDescriptorSets(dev, 5, writes, 0, ffi.NULL)
dbi2a = ffi.new("VkDescriptorBufferInfo[1]")
dbi2a[0].buffer = obuf; dbi2a[0].offset = 0; dbi2a[0].range = M*4
dbi2b = ffi.new("VkDescriptorBufferInfo[1]")
dbi2b[0].buffer = o2buf; dbi2b[0].offset = 0; dbi2b[0].range = M*4
writes2 = ffi.new("VkWriteDescriptorSet[2]")
for i in range(2):
    writes2[i].sType = vv.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
    writes2[i].dstSet = ds2[0]; writes2[i].dstBinding = i; writes2[i].dstArrayElement = 0
    writes2[i].descriptorCount = 1; writes2[i].descriptorType = 6
writes2[0].pBufferInfo = dbi2a; writes2[1].pBufferInfo = dbi2b
lib.vkUpdateDescriptorSets(dev, 2, writes2, 0, ffi.NULL)

def dispatch_once():
    r = lib.vkBeginCommandBuffer(cmdbuf[0], begin_info); assert r == 0
    lib.vkCmdBindPipeline(cmdbuf[0], 5, pipe[0])
    lib.vkCmdBindDescriptorSets(cmdbuf[0], 5, pl[0], 0, 1, ds, 0, ffi.NULL)
    pc_bytes = struct.pack("<IIf", K, NB, global_s)
    pc_c = ffi.new("char[12]")
    ffi.memmove(pc_c, pc_bytes, 12)
    lib.vkCmdPushConstants(cmdbuf[0], pl[0], 32, 0, 12, pc_c)
    lib.vkCmdDispatch(cmdbuf[0], (M + 255) // 256, 1, 1)
    r = lib.vkEndCommandBuffer(cmdbuf[0]); assert r == 0
    si = ffi.new("VkSubmitInfo*")
    si.sType = vv.VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1
    si.pCommandBuffers = cmdbuf
    lib.vkQueueSubmit(queue, 1, si, ffi.NULL)
    lib.vkDeviceWaitIdle(dev)

# ---------- 正确性 ----------
dispatch_once()
raw = map_read(omem, M*4)
gpu = list(struct.unpack("<%df" % M, raw))
bad = 0; maxerr = 0.0
for i in range(M):
    e = abs(gpu[i] - ref[i])
    maxerr = max(maxerr, e)
    if e > 1e-3 * (abs(ref[i]) + 1.0): bad += 1
print(f"correctness: M={M} K={K}  maxerr={maxerr:.5f}  bad={bad}/{M}  "
      f"({'PASS' if bad == 0 else 'FAIL'})")
print(f"  ref[0..3] = {[round(x,3) for x in ref[:4]]}")
print(f"  gpu[0..3] = {[round(x,3) for x in gpu[:4]]}")

# ---------- 带宽 ----------
wbytes = M*NB*8      # packed 权重读量
abytes = M*NB*4      # act 读量（每行读全部 act 打包）
sbytes = M*NB        # scale
asbytes = M*NB*4     # asb
total = wbytes + abytes + sbytes + asbytes
best = float("inf")
for _ in range(ITERS):
    t0 = time.perf_counter()
    dispatch_once()
    dt = time.perf_counter() - t0
    best = min(best, dt)
print(f"bw: {total/1e6:.1f} MB / {best*1000:.3f} ms = {total/best/1e9:6.2f} GB/s "
      f"(w {wbytes/1e6:.0f}MB a {abytes/1e6:.0f}MB s {sbytes/1e6:.0f}MB)")
print(f"  MAC/s: {M*K/best/1e9:7.2f} G (M={M} K={K})")
print(f"  tok/s 参考: B组权重流 {wbytes/1e6:.0f}MB/专家 → {wbytes/best/1e6:.1f} MB/ms")
