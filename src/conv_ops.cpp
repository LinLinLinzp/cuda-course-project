#include <torch/extension.h>
#include "conv.h"

void torch_launch_conv(torch::Tensor &Y,
                       const torch::Tensor &X,
                       const torch::Tensor &W,
                       int in_channels,
                       int out_channels,
                       int kernel_size,
                       int feature_size,
                       int batch_size){
    launch_conv((float *)Y.data_ptr(),
                (const float *)X.data_ptr(),
                (const float *)W.data_ptr(),
                in_channels,
                out_channels,
                kernel_size,
                feature_size,
                batch_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_launch_linear",
          &conv_launch_linear,
          "conv kernel warpper");
}
