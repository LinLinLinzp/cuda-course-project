__global__ void leakyrelu_kernel(float* output,
                                float* input,
                                float slope,
                                int dim_xw,
                                int dim_xh){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float input_s[1024];

    // if(tid < 1024){
    if(idx < dim_xw *dim_xh){
        input_s[tid] = input[idx];
    }

    if(tid < 1024){
        input_s[tid] = fmaxf(0,input_s[tid]) + slope * fminf(0,input_s[tid]);
    }

    if(idx < dim_xw *dim_xh){
        output[idx] = input_s[tid];
    }

}

void launch_leakyrelu(float* output_y,
                        float *input_X,
                        float slope,
                        int dim_xw,
                        int dim_xh
                    ){
    int num_element = dim_xw * dim_xh;

    dim3 gridSize((num_element+1023)/1024);
    dim3 blockSize(1024);

    leakyrelu_kernel<<<gridSize, blockSize>>>(output_y, \
                                            input_X, \
                                            slope, \
                                            dim_xw,\
                                            dim_xh);
}