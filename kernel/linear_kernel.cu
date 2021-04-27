__global__ void linear_kernel(float* Y,
                            const float* input_x,
                            const float* input_w,
                            int dim_xw,
                            int dim_xh,
                            int dim_ww,
                            int dim_wh
                            ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;
    // cal sum
    if(idx < dim_wh){
        for(int i = 0; i < dim_ww; i++){
            sum += input_x[i] * input_w[idx][i];
        }
    }
    // output
    if(idx < dim_wh){
        Y[idx] = sum;
    }
}

void launch_linear(float* device_y,
                    const float* input_x,
                    const float* input_w,
                    int input_dim_xw,
                    int input_dim_xh,
                    int input_dim_ww,
                    int input_dim_wh
                ){
    
    // first try batchsize = 1
    // int TILE_WIDTH = 16;

    // int dimx = (int)(ceil)((float)input_dim_ww / TILE_WIDTH);
    // int dimy = (int)(ceil)((float)input_dim_xh / TILE_WIDTH);
    
    dim3 gridSize((input_dim_wh+1023)/1024);
    dim3 blockSize(1024);

    linear_kernel<<<grid, block>>>(device_y, \
                                    input_x, \
                                    input_w, \
                                    input_dim_xw, \
                                    input_dim_xh, \
                                    input_dim_ww, \
                                    input_dim_wh
                                );
    
    
}