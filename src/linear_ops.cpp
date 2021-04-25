#include <torch/extension.h>
#include "linear.h"

void torch_launch_linear(){
    launch_linear();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_linear",
          &torch_launch_linear,
          "linear kernel warpper");
}

TORCH_LIBRARY(linear, m) {
    m.def("torch_launch_linear", torch_launch_linear);
}