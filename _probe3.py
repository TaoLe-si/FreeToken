import re
pat = r".*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$"
print("pat:", pat)
m = re.search(r"layers\.\((\d+(?:\|\d+)*)\)", pat)
print("single-backslash regex match:", m.group(1) if m else None)
m2 = re.search(r"layers.\(.+?\)", pat)
print("simple:", m2.group(0) if m2 else None)