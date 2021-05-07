__global__ void tanh_kernel(float* Y,
                            int batch_size,
                            int num){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    float tp;
    // load to register and then do tanh
    tp = Y[idx];
    Y[idx] = tanh(tp]);
}

void launch_tanh(float* Y,
                int batch_size,
                int num){
    dim3 gridSize((batch_size * num + 1023)/ 1024);
    dim3 blockSize(1024);
    tanh_kernel<<<gridSize, blockSize>>>(Y, batch_size, num);
}