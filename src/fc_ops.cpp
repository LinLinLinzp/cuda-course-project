#include <torch/extension.h>
#include "fc.h"

void torch_launch_fc(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_fc((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_fc",
          &torch_launch_fc,
          "fc2 kernel warpper");
}

TORCH_LIBRARY(fc, m) {
    m.def("torch_launch_fc", torch_launch_fc);
}