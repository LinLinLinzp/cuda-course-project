#include <torch/extension.h>
#include "leakyrelu.h"

void torch_launch_leakyrelu(torch::Tensor &output_y,
                            torch::Tensor &input_x,
                            int slope,
                            int dim_xw,
                            int dim_xh){
    launch_leakyrelu((float *)ouput_y.data_ptr(),
                    (float *)input_x.data_ptr(),
                    slope,
                    dim_xw,
                    dim_wh);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_leakyrelu",
          &torch_launch_leakyrelu,
          "leakyrelu kernel warpper");
}