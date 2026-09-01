"""Verify the bundled engine.py in FreeTokenDbg.exe."""import sys
import marshal
import zlib
import struct

EXE = r'E:\FreeToken\dist\FreeTokenDbg.exe'

# PyInstaller exe layout:
# - PKG (CArchive) at end of file
# - PYZ archive inside PKG
# - Source modules inside .pyz

with open(EXE, 'rb') as f:
    data = f.read()

# Find the PYZ archive
pyz_magic = b'PYZ\x00'
idx = data.find(pyz_magic)
if idx < 0:
    print('PYZ not found')
    sys.exit(1)
print(f'PYZ magic at offset {idx}')

# PYZ cookie structure: magic(4) + python_magic(4) + TOC offset(4) + TOC length(4) + py_version(1) + pylib_name_len(1) + pylib_name(N)
cookie = data[idx:idx+30]
print('cookie:', cookie[:25])

python_magic = struct.unpack('<I', data[idx+4:idx+8])[0]
toc_offset = struct.unpack('<I', data[idx+8:idx+12])[0]
toc_length = struct.unpack('<I', data[idx+12:idx+16])[0]
py_version = data[idx+16]
print(f'python_magic=0x{python_magic:08x} TOC offset={toc_offset} length={toc_length} py_version={py_version}')

# TOC at end of PYZ
toc_start = idx + toc_offset
toc_data = data[toc_start:toc_start+toc_length]

# Each TOC entry: 8 bytes name_len+is_code, then name
i = 0
count = 0
target_entry = None
while i < len(toc_data):
    name_len = struct.unpack('<I', toc_data[i:i+4])[0]
    is_code = toc_data[i+4]
    entry_offset = struct.unpack('<I', toc_data[i+5:i+9])[0]
    entry_length = struct.unpack('<I', toc_data[i+9:i+13])[0]
    name = toc_data[i+13:i+13+name_len].decode('utf-8')
    i += 13 + name_len
    if name == 'freetoken.engine.engine':
        target_entry = (entry_offset, entry_length, is_code)
        print(f'Found {name}: offset={entry_offset} length={entry_length} is_code={is_code}')
        break
    count += 1

if target_entry is None:
    print('engine.py NOT FOUND in PYZ')
    sys.exit(1)

# Extract the entry
offset, length, is_code = target_entry
file_offset = idx + offset
entry_data = data[file_offset:file_offset+length]

# If compressed (is_code > 0), decompress with zlib
if is_code:
    try:
        entry_data = zlib.decompress(entry_data)
        print(f'Decompressed to {len(entry_data)} bytes')
    except:
        pass

# Marshal load the code object
co = marshal.loads(entry_data)
print('Co:', co)
print('Co filename:', co.co_filename)
print('First few lines of co_consts:', co.co_consts[:3] if co.co_consts else 'empty')

# Get the source file content - check the first constant for the docstring
print('Co first const type:', type(co.co_consts[0]) if co.co_consts else 'N/A')
