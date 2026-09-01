#include <iostream>
#include <io.h>
#include <fcntl.h>
int main() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    std::cerr << "ready fd0=" << _fileno(stdin) << " fd1=" << _fileno(stdout) << " fd2=" << _fileno(stderr) << std::endl;
    char buf[8] = {0};
    int r = _read(0, buf, 8);
    std::cerr << "first read returned " << r << std::endl;
    if (r > 0) {
        std::cerr << "got: ";
        for (int i = 0; i < r; i++) std::cerr << (int)(unsigned char)buf[i] << " ";
        std::cerr << std::endl;
    }
    return 0;
}
