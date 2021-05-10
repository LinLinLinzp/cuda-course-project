#include <torch/extension.h>
#include "tanh.h"

void torch_launch_tanh(torch::Tensor &Y,
                       int batch_size,
                       int num){
    launch_tanh((float *)Y.data_ptr(),
                batch_size,
                num);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_tanh",
          &torch_launch_tanh,
          "tanh kernel warpper");
}

