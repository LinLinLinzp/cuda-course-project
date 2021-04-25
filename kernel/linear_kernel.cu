__global__ void linear_kernel(float* Y,
                            const float* input_x,
                            int dim_x
                            ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < dim_x){
        Y[idx] = input_x[idx];
    }


}

void launch_linear(float* device_y,
                    const float* input_x,
                    int input_dim_x){

    
    dim3 grid(1);
    dim3 block(16);
    linear_kernel<<<grid, block>>>(device_y, input_x, input_dim_x);
    
    
}