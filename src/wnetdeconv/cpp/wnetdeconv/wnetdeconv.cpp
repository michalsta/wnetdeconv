#include <iostream>
#include <nanobind/nanobind.h>


int hello() {
    std::cout << "Hello, world from wnetdeconv_cpp!" << std::endl;
    return 0;
}

NB_MODULE(wnetdeconv_cpp, m) {
    m.def("hello", &hello, "A function that prints 'Hello, world from wnetdeconv_cpp!'");
}