# AMDVLK 环境检测：列出可用 Vulkan ICD 与物理设备
# 用法: python amdvlk_check.py [icd_json_path]
import vulkan as vk
import os, sys

def show_devices():
    inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=b"chk", applicationVersion=1,
            apiVersion=(1 << 22) | (2 << 12))), None)
    for p in vk.vkEnumeratePhysicalDevices(inst):
        props = vk.vkGetPhysicalDeviceProperties(p)
        name = str(props.deviceName).split(chr(0))[0]
        ver = "%d.%d.%d" % ((props.apiVersion >> 22) & 0x3FF, (props.apiVersion >> 12) & 0x3FF, props.apiVersion & 0xFFF)
        drv = "%d.%d.%d" % ((props.driverVersion >> 22) & 0x3FF, (props.driverVersion >> 12) & 0x3FF, props.driverVersion & 0xFFF)
        print(f"  device: {name}  vendor=0x{props.vendorID:04X} api={ver} driver={drv}")

print("VK_ICD_FILENAMES =", os.environ.get("VK_ICD_FILENAMES", "(unset)"))
print("AMD_CONFIG_DIR   =", os.environ.get("AMD_CONFIG_DIR", "(unset)"))
icd = sys.argv[1] if len(sys.argv) > 1 else None
if icd:
    os.environ["VK_ICD_FILENAMES"] = icd
    print("using ICD:", icd)
print("physical devices:")
show_devices()
print("done")
