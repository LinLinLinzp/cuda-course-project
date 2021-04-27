#include <torch/extension.h>
#include "leakyrelu.h"

void torch_launch_leakyrelu(torch::Tensor &output_y,
                            torch::Tensor &input_x,
                            int slope){
    launch_leakyrelu((float *)ouput_y.data_ptr(),
                    (float *)input_x.data_ptr(),
                    slope);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_leakyrelu",
          &torch_launch_leakyrelu,
          "leakyrelu kernel warpper");
}