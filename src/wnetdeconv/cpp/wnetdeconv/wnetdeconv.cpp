#include <iostream>
#include <nanobind/nanobind.h>


int hello() {
    std::cout << "Hello, world from wnetdeconv_cpp!" << std::endl;
    return 0;
}

NB_MODULE(wnetdeconv_cpp, m) {
    // Build mode of *this* extension, read by is_nanobind_split() and by the
    // import-time consistency check. NB_BACKEND_MODULE is defined only when
    // nanobind_add_module() was given BACKEND_MODULE, i.e. only in split mode.
    // Extensions in different modes carry different nanobind internals and
    // silently lose sight of each other's registered types, so the mode has to
    // be observable from Python rather than inferred from a filename.
#if defined(NB_BACKEND_MODULE)
    m.attr("nanobind_split") = true;
#else
    m.attr("nanobind_split") = false;
#endif

    m.def("hello", &hello, "A function that prints 'Hello, world from wnetdeconv_cpp!'");
}