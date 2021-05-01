#include <torch/extension.h>
#include "conv.h"

void torch_launch_conv(){

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_launch_linear",
          &conv_launch_linear,
          "conv kernel warpper");
}
