// Minimal test: read 20 bytes from stdin, no _setmode
#include <iostream>
#include <io.h>
#include <fcntl.h>
int main() {
    std::cerr << "fd0=" << _fileno(stdin) << " fd1=" << _fileno(stdout) << " fd2=" << _fileno(stderr) << std::endl;
    char buf[21] = {0};
    int r = _read(0, buf, 20);
    std::cerr << "first read: " << r << " errno=" << (r<0?errno:0) << std::endl;
    if (r > 0) {
        std::cerr << "got bytes: ";
        for (int i = 0; i < r; i++) std::cerr << (int)(unsigned char)buf[i] << " ";
        std::cerr << std::endl;
    }
    // Try stdin.get
    std::cin.read(buf, 4);
    int g = (int)std::cin.gcount();
    std::cerr << "cin.read got " << g << " bytes" << std::endl;
    return 0;
}
