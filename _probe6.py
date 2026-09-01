import re
pat = r".*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$"
print("repr(pat):", repr(pat))
print("len:", len(pat))
print("char 10-30:", pat[10:30])
m = re.search(r"layers\.\(([^)]+)\)", pat)
print("v4:", m)
# 最简单的
m2 = re.search(r"layers", pat)
print("simple layers:", m2)
m3 = re.search(r"\(", pat)
print("simple paren:", m3)
m4 = re.search(r"56", pat)
print("simple 56:", m4)