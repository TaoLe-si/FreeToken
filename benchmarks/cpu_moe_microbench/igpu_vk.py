# iGPU shared-memory bandwidth via Vulkan (no shader needed: vkCmdCopyBuffer)
import vulkan as vk
import ctypes, time, sys

app = vk.VkApplicationInfo(
    sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pApplicationName=b"bw", applicationVersion=1, pEngineName=b"bw", engineVersion=1,
    apiVersion=(1<<22)|(2<<12))
info = vk.VkInstanceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, pApplicationInfo=app)
inst = vk.vkCreateInstance(info, None)
phys = vk.vkEnumeratePhysicalDevices(inst)
print("devices:", len(phys))
target = None
for i, p in enumerate(phys):
    props = vk.vkGetPhysicalDeviceProperties(p)
    name = str(props.deviceName).split("\x00")[0]
    print(f"  [{i}] {name} vendor {props.vendorID:#x} device {props.deviceID:#x}")
    if props.vendorID == 0x1002:  # AMD
        target = p
if target is None:
    raise SystemExit("no AMD device")
props = vk.vkGetPhysicalDeviceProperties(target)
devname = str(props.deviceName).split("\x00")[0]
print("using:", devname)

# queue family (prefer compute-capable)
qprops = vk.vkGetPhysicalDeviceQueueFamilyProperties(target)
qf = None
for i, q in enumerate(qprops):
    if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
        qf = i; break
if qf is None:
    for i, q in enumerate(qprops):
        if q.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
            qf = i; break
print("queue family:", qf)
prio = [1.0]
qci = vk.VkDeviceQueueCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                 queueFamilyIndex=qf, queueCount=1, pQueuePriorities=prio)
dci = vk.VkDeviceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            queueCreateInfoCount=1, pQueueCreateInfos=[qci])
dev = vk.vkCreateDevice(target, dci, None)
queue = vk.vkGetDeviceQueue(dev, qf, 0)

# host-visible memory type
memprops = vk.vkGetPhysicalDeviceMemoryProperties(target)
hv_type = None
for i, mt in enumerate(memprops.memoryTypes):
    if mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT and \
       mt.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:
        hv_type = i; break
print("host-visible mem type:", hv_type)

def make_buf(size):
    bci = vk.VkBufferCreateInfo(sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                size=size, usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE)
    buf = vk.vkCreateBuffer(dev, bci, None)
    req = vk.vkGetBufferMemoryRequirements(dev, buf)
    aci = vk.VkMemoryAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                  allocationSize=req.size, memoryTypeIndex=hv_type)
    mem = vk.vkAllocateMemory(dev, aci, None)
    vk.vkBindBufferMemory(dev, buf, mem, 0)
    return buf, mem

# command pool + buffers
cpi = vk.VkCommandPoolCreateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                 flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex=qf)
pool = vk.vkCreateCommandPool(dev, cpi, None)
abci = vk.VkCommandBufferAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                      commandPool=pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1)
cmdbuf = vk.vkAllocateCommandBuffers(dev, abci)[0]

def bench(gb, iters=5):
    size = gb << 30
    bufA, memA = make_buf(size)
    bufB, memB = make_buf(size)
    
    best = float("inf")
    for _ in range(iters):
        bci = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO)
        vk.vkBeginCommandBuffer(cmdbuf, bci)
        reg = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=size)
        vk.vkCmdCopyBuffer(cmdbuf, bufA, bufB, 1, [reg])
        vk.vkEndCommandBuffer(cmdbuf)
        si = vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=[cmdbuf])
        t0 = time.perf_counter()
        vk.vkQueueSubmit(queue, 1, [si], None)
        vk.vkDeviceWaitIdle(dev)
        best = min(best, time.perf_counter() - t0)
    gbps = size / best / 1e9
    print(f"copy {gb:2d}GB (host-visible shared): {gbps:7.1f} GB/s  (read-equivalent ~{gbps*2:6.1f})")
    vk.vkFreeMemory(dev, memA, None); vk.vkFreeMemory(dev, memB, None)
    vk.vkDestroyBuffer(dev, bufA, None); vk.vkDestroyBuffer(dev, bufB, None)

bench(1)
bench(2)
bench(4)
