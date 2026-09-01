import re
pat = r".*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$"
print("pat:", pat)
# 测试多个变体
m = re.search(r"layers\.\((\d+(?:|\d+)*)\)", pat)
print("v1:", m.group(1) if m else None)
m = re.search(r"layers\.\((\d+(?:\d+|)*\d+)\)", pat)
print("v2:", m.group(1) if m else None)
m = re.search(r"layers\.\((\d+(?:\d+|)*)\)", pat)
print("v3:", m.group(1) if m else None)
# 直接贪婪匹配到右括号
m = re.search(r"layers\.\(([^)]+)\)", pat)
print("v4:", m.group(1) if m else None)