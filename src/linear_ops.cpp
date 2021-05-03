#include <torch/extension.h>
#include "linear.h"

void torch_launch_linear(torch::Tensor &device_y,
                        const torch::Tensor &input_x,
                        const torch::Tensor &input_w,
                        int input_dim_xw,
                        int input_dim_xh,
                        int input_dim_ww,
                        int input_dim_wh
){
    launch_linear((float *)device_y.data_ptr(),
                  (const float *)input_x.data_ptr(),
                  (const float *)input_w.data_ptr(),
                    input_dim_xw,
                    input_dim_xh,
                    input_dim_ww,
                    input_dim_wh
                );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_linear",
          &torch_launch_linear,
          "linear kernel warpper");
}

// TORCH_LIBRARY(linear, m) {
//     m.def("torch_launch_linear", torch_launch_linear);
// }