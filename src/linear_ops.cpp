#include <torch/extension.h>
#include "linear.h"

void torch_launch_linear(torch::Tensor &device_y,
                        const torch::Tensor &input_x,
                        int input_dim_x
){
    launch_linear((float *)device_y.data_ptr(),
                  (const float*)input_x.data_ptr(),
                  input_dim_x
                );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_linear",
          &torch_launch_linear,
          "linear kernel warpper");
}

TORCH_LIBRARY(linear, m) {
    m.def("torch_launch_linear", torch_launch_linear);
}