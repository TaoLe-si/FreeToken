# 真实任务链：GEMV_A(W4A8) → 激活量化 → GEMV_B(W4A8) → CPU 消费
# 模拟 MoE 专家推理：中间激活跨 shader 消费（LLPC 不应消除）
# 用法: python igpu_gemv_chain.py [M1] [K1] [M2] [iters]
import vulkan as vk
import vulkan._vulkan as vv
import time, struct, os, sys, random, math
ffi = vv.ffi
lib = vv.lib

M1 = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
K1 = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
M2 = int(sys.argv[3]) if len(sys.argv) > 3 else 512
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
NB1, NB2 = K1 // 16, M1 // 16
NQ1 = M1 // 16   # quant 块数（GEMV_A 输出 M1 维）
DIR = os.path.dirname(os.path.abspath(__file__))

want = int(os.environ.get("VK_DEV", "0x1002"), 16)
app = vk.VkApplicationInfo(sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName=b"chain", applicationVersion=1, pEngineName=b"chain", engineVersion=1,
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
dev = vk.vkCreateDevice(target, vk.VkDeviceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    queueCreateInfoCount=1, pQueueCreateInfos=[vk.VkDeviceQueueCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, queueFamilyIndex=qf,
        queueCount=1, pQueuePriorities=[1.0])]), None)
queue = vk.vkGetDeviceQueue(dev, qf, 0)
memprops = vk.vkGetPhysicalDeviceMemoryProperties(target)
hv = next(i for i, mt in enumerate(memprops.memoryTypes)
          if mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT and
             mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

def load_spv(name):
    spv = open(os.path.join(DIR, name), "rb").read()
    w = ffi.new("uint32_t[]", list(struct.unpack("<%dI" % (len(spv)//4), spv)))
    sci = ffi.new("VkShaderModuleCreateInfo*")
    sci.sType = vv.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    sci.flags = 0; sci.codeSize = len(spv); sci.pCode = w
    sm = ffi.new("VkShaderModule*")
    assert lib.vkCreateShaderModule(dev, sci, ffi.NULL, sm) == 0
    return sm

def mk_pipe(sm, nb_bind, pc_size):
    bindings = ffi.new("VkDescriptorSetLayoutBinding[%d]" % nb_bind)
    for bi in range(nb_bind):
        bindings[bi].binding = bi; bindings[bi].descriptorType = 6
        bindings[bi].descriptorCount = 1; bindings[bi].stageFlags = 32
        bindings[bi].pImmutableSamplers = ffi.NULL
    dsl_info = ffi.new("VkDescriptorSetLayoutCreateInfo*")
    dsl_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    dsl_info.flags = 0; dsl_info.bindingCount = nb_bind; dsl_info.pBindings = bindings
    dsl = ffi.new("VkDescriptorSetLayout*")
    assert lib.vkCreateDescriptorSetLayout(dev, dsl_info, ffi.NULL, dsl) == 0
    pc = ffi.new("VkPushConstantRange[1]")
    pc[0].stageFlags = 32; pc[0].offset = 0; pc[0].size = pc_size
    pl_info = ffi.new("VkPipelineLayoutCreateInfo*")
    pl_info.sType = vv.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    pl_info.flags = 0; pl_info.setLayoutCount = 1; pl_info.pSetLayouts = dsl
    pl_info.pushConstantRangeCount = 1; pl_info.pPushConstantRanges = pc
    pl = ffi.new("VkPipelineLayout*")
    assert lib.vkCreatePipelineLayout(dev, pl_info, ffi.NULL, pl) == 0
    name = ffi.new("char[]", b"main")
    cpi = ffi.new("VkComputePipelineCreateInfo*")
    cpi.sType = vv.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    cpi.pNext = ffi.NULL; cpi.flags = 0; cpi.layout = pl[0]
    cpi.basePipelineHandle = ffi.NULL; cpi.basePipelineIndex = -1
    cpi.stage.sType = vv.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
    cpi.stage.pNext = ffi.NULL; cpi.stage.flags = 0
    cpi.stage.stage = 32; cpi.stage.module = sm[0]
    cpi.stage.pName = name; cpi.stage.pSpecializationInfo = ffi.NULL
    pipe = ffi.new("VkPipeline*")
    assert lib.vkCreateComputePipelines(dev, ffi.NULL, 1, cpi, ffi.NULL, pipe) == 0
    # descriptor pool + set
    psize = ffi.new("VkDescriptorPoolSize[1]")
    psize[0].type = 6; psize[0].descriptorCount = nb_bind
    pool_info = ffi.new("VkDescriptorPoolCreateInfo*")
    pool_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    pool_info.flags = 0; pool_info.maxSets = 1; pool_info.poolSizeCount = 1
    pool_info.pPoolSizes = psize
    pool = ffi.new("VkDescriptorPool*")
    assert lib.vkCreateDescriptorPool(dev, pool_info, ffi.NULL, pool) == 0
    ds_info = ffi.new("VkDescriptorSetAllocateInfo*")
    ds_info.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    ds_info.descriptorPool = pool[0]; ds_info.descriptorSetCount = 1
    ds_info.pSetLayouts = dsl
    ds = ffi.new("VkDescriptorSet*")
    assert lib.vkAllocateDescriptorSets(dev, ds_info, ds) == 0
    return pl[0], pipe[0], ds[0]

sm_gemv = load_spv("igpu_gemv.spv")
sm_quant = load_spv("igpu_quant.spv")
pl_g, pipe_g, ds_gA = mk_pipe(sm_gemv, 5, 12)   # gemv：K, nb, global
pl_q, pipe_q, ds_q = mk_pipe(sm_quant, 3, 4)     # quant：N
# 第二个 gemv set（gemv_B 用不同 buffer）
pl_g2, pipe_g2, ds_gB = mk_pipe(sm_gemv, 5, 12)

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

def make_buf(size):
    bci = vk.VkBufferCreateInfo(sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        size=size, usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE)
    buf = vk.vkCreateBuffer(dev, bci, None)
    req = vk.vkGetBufferMemoryRequirements(dev, buf)
    mem = vk.vkAllocateMemory(dev, vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=req.size, memoryTypeIndex=hv), None)
    vk.vkBindBufferMemory(dev, buf, mem, 0)
    return buf, mem

def map_write(mem, b):
    ptr = ffi.new("void**")
    assert lib.vkMapMemory(dev, mem, 0, len(b), 0, ptr) == 0
    ffi.memmove(ptr[0], b, len(b))
    lib.vkUnmapMemory(dev, mem)

def map_read(mem, n):
    ptr = ffi.new("void**")
    assert lib.vkMapMemory(dev, mem, 0, n, 0, ptr) == 0
    b = ffi.buffer(ptr[0], n)[:]
    lib.vkUnmapMemory(dev, mem)
    return b

def upd_desc(ds, bufs):
    n = len(bufs)
    dbis = [ffi.new("VkDescriptorBufferInfo[1]") for _ in range(n)]
    for i, (b, sz) in enumerate(bufs):
        dbis[i][0].buffer = b; dbis[i][0].offset = 0; dbis[i][0].range = sz
    ws = ffi.new("VkWriteDescriptorSet[%d]" % n)
    for i in range(n):
        ws[i].sType = vv.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        ws[i].dstSet = ds; ws[i].dstBinding = i; ws[i].dstArrayElement = 0
        ws[i].descriptorCount = 1; ws[i].descriptorType = 6
        ws[i].pBufferInfo = dbis[i]
    lib.vkUpdateDescriptorSets(dev, n, ws, 0, ffi.NULL)

def dispatch(pipeline, layout, ds, pc_bytes, groups):
    r = lib.vkBeginCommandBuffer(cmdbuf[0], begin_info); assert r == 0
    lib.vkCmdBindPipeline(cmdbuf[0], 5, pipeline)
    lib.vkCmdBindDescriptorSets(cmdbuf[0], 5, layout, 0, 1, ffi.new("VkDescriptorSet[1]", [ds]), 0, ffi.NULL)
    pc_c = ffi.new("char[%d]" % len(pc_bytes))
    ffi.memmove(pc_c, pc_bytes, len(pc_bytes))
    lib.vkCmdPushConstants(cmdbuf[0], layout, 32, 0, len(pc_bytes), pc_c)
    lib.vkCmdDispatch(cmdbuf[0], groups, 1, 1)
    r = lib.vkEndCommandBuffer(cmdbuf[0]); assert r == 0
    si = ffi.new("VkSubmitInfo*")
    si.sType = vv.VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1
    si.pCommandBuffers = cmdbuf
    lib.vkQueueSubmit(queue, 1, si, ffi.NULL)
    lib.vkDeviceWaitIdle(dev)

# ---------- 数据（真实任务：随机权重银行 + 输入激活）----------
random.seed(7)
K_E2M1X2 = [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12]
def e4m3(i):
    s = -1.0 if (i & 0x80) else 1.0
    e = (i >> 3) & 0xF; m = i & 7
    return s * (m * 0.001953125 if e == 0 else (1.0 + m/8.0) * 2.0**(e-7))

def gen_w(M, K):
    nb = K // 16
    packed = [random.randrange(256) for _ in range(M * nb * 8)]
    scale = [random.randrange(256) for _ in range(M * nb)]
    return packed, scale

def pack_act(acts):
    # acts: float list（K 维）→ 打包 int8（每块 [e0..3][e4..7][o0..3][o4..7]）
    nb = len(acts) // 16
    out = []
    for b in range(nb):
        ev = [int(max(-127, min(127, round(v)))) for v in acts[b*16:b*16+8]]
        od = [int(max(-127, min(127, round(v)))) for v in acts[b*16+8:b*16+16]]
        for grp in (ev[0:4], ev[4:8], od[0:4], od[4:8]):
            u = 0
            for j in range(4): u |= (grp[j] & 0xFF) << (j*8)
            out.append(u)
    return out

# GEMV_A：输入激活 x（int8 随机 -127..127）+ asb
x_act = [random.randrange(255) - 127 for _ in range(K1)]
x_asb = [0.01 + 0.05 * random.randrange(100)/100.0 for _ in range(NB1)]
W1p, W1s = gen_w(M1, K1)
W2p, W2s = gen_w(M2, M1)   # GEMV_B：M2×M1

# CPU 参考：完整链
def cpu_gemv_row(packed, scale, acts, asbs, row, K, global_s):
    nb = K // 16
    acc = 0.0
    wbase = row * nb * 8; sbase = row * nb
    for b in range(nb):
        isum = 0
        for j in range(8):
            pk = packed[wbase + b*8 + j]
            isum += K_E2M1X2[pk & 0xF] * acts[b*16+j] + K_E2M1X2[pk >> 4] * acts[b*16+8+j]
        acc += e4m3(scale[sbase+b]) * asbs[b] * isum
    return acc * global_s * 0.5

GLOBAL = 0.25
# 阶段1 CPU 参考
y1_ref = [cpu_gemv_row(W1p, W1s, x_act, x_asb, r, K1, GLOBAL) for r in range(M1)]
# 阶段2 quant CPU 参考
# 简化：直接生成 y1q 参考（sc 用块内 absmax/127）
y1q = [0]*M1; asb1 = [0.0]*NQ1
for b in range(NQ1):
    blk = y1_ref[b*16:(b+1)*16]
    am = max(abs(v) for v in blk)
    sc = am/127.0 if am > 0 else 1.0
    asb1[b] = sc
    for j in range(16):
        q = int(round(blk[j]/sc))
        y1q[b*16+j] = max(-127, min(127, q))
y2_ref = [cpu_gemv_row(W2p, W2s, y1q, asb1, r, M1, GLOBAL) for r in range(M2)]

# ---------- GPU buffers ----------
w1b, w1m = make_buf(M1*NB1*8); s1b, s1m = make_buf(M1*NB1)
xab, xam = make_buf(NB1*4*4); xsb, xsm = make_buf(NB1*4)
y1b, y1m = make_buf(M1*4)                       # GEMV_A 输出（float）
y1ib, y1im = make_buf(NQ1*4*4); y1sb, y1sm = make_buf(NQ1*4)  # quant 输出
w2b, w2m = make_buf(M2*NB2*8); s2b, s2m = make_buf(M2*NB2)
y2b, y2m = make_buf(M2*4)                       # GEMV_B 输出（float，CPU 消费）
map_write(w1m, bytes(W1p)); map_write(s1m, bytes(W1s))
map_write(xam, struct.pack("<%dI" % (NB1*4), *pack_act(x_act)))
map_write(xsm, struct.pack("<%df" % NB1, *x_asb))
map_write(y1im, struct.pack("<%dI" % (NQ1*4), *([0]*(NQ1*4))))
map_write(y1sm, struct.pack("<%df" % NQ1, *([0.0]*NQ1)))
map_write(w2m, bytes(W2p)); map_write(s2m, bytes(W2s))
map_write(y1m, struct.pack("<%dI" % M1, *([0x5A5A5A5A]*M1)))  # 预填充检测
map_write(y2m, struct.pack("<%df" % M2, *([0.0]*M2)))

upd_desc(ds_gA, [(w1b, M1*NB1*8), (s1b, M1*NB1), (xab, NB1*4*4), (xsb, NB1*4), (y1b, M1*4)])
upd_desc(ds_q,  [(y1b, M1*4), (y1ib, NQ1*4*4), (y1sb, NQ1*4)])
upd_desc(ds_gB, [(w2b, M2*NB2*8), (s2b, M2*NB2), (y1ib, NQ1*4*4), (y1sb, NQ1*4), (y2b, M2*4)])

# ---------- 链执行 ----------
def run_chain():
    dispatch(pipe_g, pl_g, ds_gA, struct.pack("<IIf", K1, NB1, GLOBAL), (M1+255)//256)
    dispatch(pipe_q, pl_q, ds_q, struct.pack("<I", M1), (NQ1+255)//256)
    dispatch(pipe_g2, pl_g2, ds_gB, struct.pack("<IIf", M1, NQ1, GLOBAL), (M2+255)//256)

run_chain()
y2 = list(struct.unpack("<%df" % M2, map_read(y2m, M2*4)))
bad = 0; maxerr = 0.0
for i in range(M2):
    e = abs(y2[i] - y2_ref[i]); maxerr = max(maxerr, e)
    if e > 1e-2 * (abs(y2_ref[i]) + 1.0): bad += 1
print(f"chain correctness: bad={bad}/{M2} maxerr={maxerr:.4f} ({'PASS' if bad==0 else 'FAIL'})")
print(f"  y2[0..3] ref = {[round(v,2) for v in y2_ref[:4]]}")
print(f"  y2[0..3] gpu = {[round(v,2) for v in y2[:4]]}")
y1raw = map_read(y1m, M1*4)
print("  y1 raw[0..3] =", [hex(struct.unpack("<I", y1raw[i*4:(i+1)*4])[0]) for i in range(4)])
y1g = list(struct.unpack("<%df" % M1, map_read(y1m, M1*4)))
b1 = sum(1 for i in range(M1) if abs(y1g[i] - y1_ref[i]) > 1e-2*(abs(y1_ref[i])+1.0))
print(f"  stage1 (GEMV_A) bad={b1}/{M1}  stage1 非零值: {sum(1 for v in y1g if v != 0.0)}/{M1}")

# ---------- 分阶段计时 ----------
def time_it(fn):
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter(); fn(); dt = time.perf_counter() - t0
        best = min(best, dt)
    return best

tA = time_it(lambda: dispatch(pipe_g, pl_g, ds_gA, struct.pack("<IIf", K1, NB1, GLOBAL), (M1+255)//256))
tQ = time_it(lambda: dispatch(pipe_q, pl_q, ds_q, struct.pack("<I", M1), (NQ1+255)//256))
tB = time_it(lambda: dispatch(pipe_g2, pl_g2, ds_gB, struct.pack("<IIf", M1, NQ1, GLOBAL), (M2+255)//256))
wA = M1*NB1*8 + M1*NB1 + NB1*4*4 + NB1*4
wB = M2*NB2*8 + M2*NB2 + NB1*4*4 + NB1*4
print(f"stage A (GEMV_A M={M1} K={K1}): {tA*1000:.3f} ms  → 读 {wA/1e6:.1f} MB = {wA/tA/1e9:.1f} GB/s  {M1*K1/tA/1e9:.1f} G MAC/s")
print(f"stage Q (quant M={M1}):     {tQ*1000:.3f} ms  → {M1*4/tQ/1e9:.1f} GB/s")
print(f"stage B (GEMV_B M={M2} K={M1}): {tB*1000:.3f} ms  → 读 {wB/1e6:.1f} MB = {wB/tB/1e9:.1f} GB/s  {M2*M1/tB/1e9:.1f} G MAC/s")
tchain = time_it(run_chain)
print(f"chain total: {tchain*1000:.3f} ms  → {M2*M1/tchain/1e9:.1f} G MAC/s (链端到端)")
print(f"参考: copy 引擎 28.4 GB/s → B 组权重流 {wA/1e6:.1f} MB/专家 → {(wA/1e6)/28.4:.2f} ms/专家（copy 方案）")
