#include <torch/extension.h>
#include "convtranspose.h"

void torch_launch_convtranspose_1(torch::Tensor &Y,
                                  const torch::Tensor &X,
                                  const torch::Tensor &W,
                                  int in_channels,
                                  int out_channels,
                                  int kernel_size,
                                  int feature_size,
                                  int batch_size,
                                  int stride){
    torch_launch_convtranspose_1((float *)Y.data_ptr(),
                                 (const float *)X.data_ptr(),
                                 (const float *)W.data_ptr(),
                                 in_channels,
                                 out_channels,
                                 kernel_size,
                                 feature_size,
                                 batch_size,
                                 stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("torch_launch_convtranspose_1",
          &torch_launch_convtranspose_1,
          "convtranspose1 kernel warpper");
}
