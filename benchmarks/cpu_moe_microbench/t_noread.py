import vulkan as vk
import vulkan._vulkan as vv
import time, struct, os
ffi = vv.ffi
lib = vv.lib
DIR = os.path.dirname(os.path.abspath(__file__))
want = int(os.environ.get("VK_DEV", "0x1002"), 16)
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pApplicationInfo=vk.VkApplicationInfo(sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName=b"t", applicationVersion=1, pEngineName=b"t", engineVersion=1,
        apiVersion=(1 << 22) | (2 << 12))), None)
phys = vk.vkEnumeratePhysicalDevices(inst)
target = next(p for p in phys if vk.vkGetPhysicalDeviceProperties(p).vendorID == want)
print("device:", str(vk.vkGetPhysicalDeviceProperties(target).deviceName).split(chr(0))[0])
qprops = vk.vkGetPhysicalDeviceQueueFamilyProperties(target)
qf = next(i for i, q in enumerate(qprops) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)
dev = vk.vkCreateDevice(target, vk.VkDeviceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    queueCreateInfoCount=1, pQueueCreateInfos=[vk.VkDeviceQueueCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, queueFamilyIndex=qf,
        queueCount=1, pQueuePriorities=[1.0])]), None)
queue = vk.vkGetDeviceQueue(dev, qf, 0)
mp = vk.vkGetPhysicalDeviceMemoryProperties(target)
hv = next(i for i, mt in enumerate(mp.memoryTypes)
          if mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT and
             mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
import sys
spvname = sys.argv[1] if len(sys.argv) > 1 else "igpu_noread.spv"
spv = open(os.path.join(DIR, spvname), "rb").read()
w = ffi.new("uint32_t[]", list(struct.unpack("<%dI" % (len(spv)//4), spv)))
sci = ffi.new("VkShaderModuleCreateInfo*")
sci.sType = vv.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
sci.flags = 0; sci.codeSize = len(spv); sci.pCode = w
sm = ffi.new("VkShaderModule*")
assert lib.vkCreateShaderModule(dev, sci, ffi.NULL, sm) == 0
b = ffi.new("VkDescriptorSetLayoutBinding[1]")
b[0].binding = 0; b[0].descriptorType = 6; b[0].descriptorCount = 1
b[0].stageFlags = 32; b[0].pImmutableSamplers = ffi.NULL
dli = ffi.new("VkDescriptorSetLayoutCreateInfo*")
dli.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
dli.flags = 0; dli.bindingCount = 1; dli.pBindings = b
dsl = ffi.new("VkDescriptorSetLayout*")
assert lib.vkCreateDescriptorSetLayout(dev, dli, ffi.NULL, dsl) == 0
pli = ffi.new("VkPipelineLayoutCreateInfo*")
pli.sType = vv.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
pli.flags = 0; pli.setLayoutCount = 1; pli.pSetLayouts = dsl
pl = ffi.new("VkPipelineLayout*")
assert lib.vkCreatePipelineLayout(dev, pli, ffi.NULL, pl) == 0
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
ps = ffi.new("VkDescriptorPoolSize[1]"); ps[0].type = 6; ps[0].descriptorCount = 1
pi2 = ffi.new("VkDescriptorPoolCreateInfo*")
pi2.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
pi2.flags = 0; pi2.maxSets = 1; pi2.poolSizeCount = 1; pi2.pPoolSizes = ps
pool = ffi.new("VkDescriptorPool*")
assert lib.vkCreateDescriptorPool(dev, pi2, ffi.NULL, pool) == 0
dsi = ffi.new("VkDescriptorSetAllocateInfo*")
dsi.sType = vv.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
dsi.descriptorPool = pool[0]; dsi.descriptorSetCount = 1; dsi.pSetLayouts = dsl
ds = ffi.new("VkDescriptorSet*")
assert lib.vkAllocateDescriptorSets(dev, dsi, ds) == 0
cpi2 = ffi.new("VkCommandPoolCreateInfo*")
cpi2.sType = vv.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
cpi2.flags = vv.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi2.queueFamilyIndex = qf
pool2 = ffi.new("VkCommandPool*")
assert lib.vkCreateCommandPool(dev, cpi2, ffi.NULL, pool2) == 0
abi = ffi.new("VkCommandBufferAllocateInfo*")
abi.sType = vv.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
abi.commandPool = pool2[0]; abi.level = 0; abi.commandBufferCount = 1
cb = ffi.new("VkCommandBuffer*")
assert lib.vkAllocateCommandBuffers(dev, abi, cb) == 0
bi2 = ffi.new("VkCommandBufferBeginInfo*")
bi2.sType = vv.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; bi2.flags = 0
# buffer 1MB 预填充 0x5A5A5A5A
N = 1 << 20
bci = vk.VkBufferCreateInfo(sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    size=N*4, usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE)
buf = vk.vkCreateBuffer(dev, bci, None)
req = vk.vkGetBufferMemoryRequirements(dev, buf)
mem = vk.vkAllocateMemory(dev, vk.VkMemoryAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    allocationSize=req.size, memoryTypeIndex=hv), None)
vk.vkBindBufferMemory(dev, buf, mem, 0)
ptr = ffi.new("void**")
assert lib.vkMapMemory(dev, mem, 0, N*4, 0, ptr) == 0
ffi.memmove(ptr[0], struct.pack("<%dI" % N, *([0x5A5A5A5A]*N)), N*4)
lib.vkUnmapMemory(dev, mem)
dbi = ffi.new("VkDescriptorBufferInfo[1]")
dbi[0].buffer = buf; dbi[0].offset = 0; dbi[0].range = N*4
ws = ffi.new("VkWriteDescriptorSet[1]")
ws[0].sType = vv.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
ws[0].dstSet = ds[0]; ws[0].dstBinding = 0; ws[0].dstArrayElement = 0
ws[0].descriptorCount = 1; ws[0].descriptorType = 6; ws[0].pBufferInfo = dbi
lib.vkUpdateDescriptorSets(dev, 1, ws, 0, ffi.NULL)
def run_once():
    assert lib.vkBeginCommandBuffer(cb[0], bi2) == 0
    lib.vkCmdBindPipeline(cb[0], 5, pipe[0])
    lib.vkCmdBindDescriptorSets(cb[0], 5, pl[0], 0, 1, ds, 0, ffi.NULL)
    lib.vkCmdDispatch(cb[0], N//256, 1, 1)
    assert lib.vkEndCommandBuffer(cb[0]) == 0
    si = ffi.new("VkSubmitInfo*")
    si.sType = vv.VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1
    si.pCommandBuffers = cb
    lib.vkQueueSubmit(queue, 1, si, ffi.NULL)
    lib.vkDeviceWaitIdle(dev)
run_once()
ptr2 = ffi.new("void**")
assert lib.vkMapMemory(dev, mem, 0, 16, 0, ptr2) == 0
raw = ffi.buffer(ptr2[0], 16)[:]
lib.vkUnmapMemory(dev, mem)
print("noread c_blob[0..3] =", [hex(struct.unpack("<I", raw[i*4:(i+1)*4])[0]) for i in range(4)])
# 时间
best = float("inf")
for _ in range(10):
    t0 = time.perf_counter(); run_once(); dt = time.perf_counter() - t0
    best = min(best, dt)
print(f"noread time: {best*1000:.3f} ms (N={N}) → {N*4/best/1e9:.1f} GB/s 写")
