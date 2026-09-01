import re
pat = r".*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$"
print("repr:", repr(pat))
m = re.search(r"layers\.\(([^)]+)\)", pat)
print("v:", m.group(1) if m else None)
