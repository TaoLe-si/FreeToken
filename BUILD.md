# FreeToken.exe 编译说明

## 环境要求

- **Windows 10/11 (x64)**
- **Python 3.12** (CPython, 64-bit) — 通过 uv 安装到 venv
- **PyTorch 2.11+ / 2.12**（必须与 FreeToken wheel 兼容）

## Python venv 位置

两个 venv，分别用于不同目的：

| 路径 | 用途 |
|---|---|
| `C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\python.exe` | dev daemon / 运行 cli |
| `C:\Users\Administrator\AppData\Local\FreeToken\venv\Scripts\python.exe` | 构建 C++ 扩展（PyTorch 头文件） |

## 编译命令

### 1. 生产版（窗口模式，console=False）

```bat
cd /d E:\FreeToken
C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\pyinstaller.exe --clean --noconfirm FreeToken.spec
```

输出：`E:\FreeToken\dist\FreeToken.exe`

### 2. 调试版（控制台模式，console=True）

```bat
cd /d E:\FreeToken
C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\pyinstaller.exe --clean --noconfirm FreeTokenDbg.spec
```

输出：`E:\FreeToken\dist\FreeTokenDbg.exe`

## Spec 文件说明

### `FreeToken.spec`
- `console=False`（窗口模式，无控制台窗口）
- 打包内核扩展（_freetoken_igpu.pyd, _pinned_tensor.pyd）
- 默认用于发布

### `FreeTokenDbg.spec`
- `console=True`（带控制台，可看 stdout/stderr）
- 不打包内核扩展（直接用 source 树内的 .pyd）
- 用于本地调试

两个 spec 都引用 `launcher.py` 作为入口。

## 编译后产物

```
E:\FreeToken\dist\FreeToken.exe          ~50MB（生产版单文件）
E:\FreeToken\dist\FreeTokenDbg.exe       ~50MB（调试版单文件）
E:\FreeToken\dist\freeToken\             -- packed data dir (freetoken module)
E:\FreeToken\dist\bin\                  -- bundled binaries
```

## 调试

运行调试版时，会弹出 console 窗口，所有 `print()` 和 `logger.info()` 输出可见。

调试时建议：
1. 用 `FreeTokenDbg.exe`（console=True）
2. 关注以下日志标记定位问题：
   - `[dbg-eng]` — engine 前后向调用栈
   - `[dbg-fwd]` — scheduler 前后向调用栈
   - `[dbg-reply]` — DetokenizeMsg 发送点
   - `[dbg-proc]` — _process_last_data 入口
   - `[MTP-dbg]` — MTP 验证批次详情（需设环境变量 `FT_MTP_DEBUG=1`）

## 常见问题

### Q: 编译报错 "ModuleNotFoundError: No module named 'pyinstaller'"
A: 在 venv 中安装：
```bat
C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\pip.exe install pyinstaller
```

### Q: 编译报错 "_freetoken_igpu.pyd not found"
A: 需先运行 `_build_glue.bat` 编译 C++ 扩展。

### Q: 编译成功但启动后崩溃
A: 检查 `dist\freeToken\python\freetoken\kernel\\_freetoken_igpu.cp312-win_amd64.pyd` 是否存在。如果 `FreeToken.spec` 用二进制打包模式（`binaries=[...]`），必须先编译 .pyd。

## 完整编译流程（从零开始）

```bat
REM 1. 编译 C++ 扩展
cd /d E:\FreeToken
.\_build_glue.bat

REM 2. 编译 MTP 服务器扩展（如需要）
.\\_build_mtp_servers.bat

REM 3. 编译 exe（生产版 + 调试版）
C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\pyinstaller.exe --clean --noconfirm FreeToken.spec
C:\Users\Administrator\AppData\Local\freeToken\venv\Scripts\pyinstaller.exe --clean --noconfirm FreeTokenDbg.spec

REM 4. 测试
.\dist\FreeTokenDbg.exe
```
