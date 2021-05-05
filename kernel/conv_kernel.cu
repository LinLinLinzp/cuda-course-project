__global__ void conv_kernel_exp(float* Y,
                            const float* X,
                            const float* W,
                            int in_channels,
                            int out_channels,
                            int kernel_size,
                            int feature_size,
                            int batch_size){
    // X: [1, 256, 7, 7]
    // batch x in_channel x kernel_size x kernel_size
    // W: [128, 256, 5, 5]
    //  out_channel x in_channels x kernel_size x kernel_size
    // Y: [1, 128, 7, 7]
    // batch x out_channels x kernel_size x kernel_size
    
    int batch,out,h,w,in_channel_size,X_base;

    batch = blockIdx.x;
    out = blockIdx.y;
    h = threadIdx.y;
    w = threadIdx.x;
    in_channel_size = in_channels * kernel_size * kernel_size;

    // float sum = 0.;
    // for(int in = 0; in < in_channels; in ++){
    //     for(int p = 0; p < kernel_size; p++){
    //         for(int q = 0; q < kernel_size; q++){
    //             X_idx = batch * in_channel_size + in * kernel_size * kernel_size + h * kernel_size + w;
    //             W_idx = 
    //         }
    //     }
    // }
    
    __shared__ float shared_X[7+4][7+4];
    __shared__ float shared_W[5][5];
    
    // // load to shared memory
    if(batch < batch_size){ //for each batch, current = 1
        
        if(out < out_channels){ //for each out channels
            for (int in = 0; in < in_channels; in++){
                X_idx = batch * in_channel_size + in * kernel_size * kernel_size + h * kernel_size + w;
                W_idx = 

            }

            
            
        }
    }
    
    // float sum = 0.;
    // for (int in = 0; in < in_channels; in++){
    //     for(int i = 0; i < kernel_size; i++){
    //         for(int j = 0; j < kernel_size; j++){
    //             idx_X_start = batch * in_channel_size + in * kernel_size * kernel_size;
    //             idx_W_start = out * in_channel_size + in * kernel_size * kernel_size;
    //             idx_X = idx_X_start + h * w;

    //             sum += X[]
    //         }
    //     }
    // }




    // int in = blockIdx.x;
    // int out = blockIdx.y;
    // int h = threadIdx.y;
    // int w = threadIdx.x;

    // // int batch_dim = kernel_size * kernel_size 

    
    // for (int batch = 0; batch < batch_size; batch ++){
    //     float sum = 0.;
    //     if (in < in_channels, out < out_channels){
    //         for (int idx_k = 0; idx_k < kernel_size * kernel_size; idx_k++){
    //             idx_X = batch * in_channels * kernel_size * kernel_size + in;
    //             idx_W = 
    //         }
    //     }

    // }


    
    // for(int batch = 0; batch < batch_size; batch++){
        // for (int out = 0; out < out_channels; out++){
            
        // }
    // }
}
__global__ void conv_kernel(float* Y,
                            const float* X,
                            const float* W,
                            int in_channels,
                            int out_channels,
                            int kernel_size,
                            int feature_size,
                            int batch_size){
    // X: [1, 256, 7, 7]
    // batch x in_channel x feature_size x feature_size
    // W: [128, 256, 5, 5]
    //  out_channel x in_channels x kernel_size x kernel_size
    // Y: [1, 128, 7, 7]
    // batch x out_channels x feature_size x feature_size

    int batch, out;

    __shared__ float shared_X[7+2][7+2];
    __shared__ float shared_W[5][5];
    // __shared__ float shared_Y[7][7];

    batch = blockIdx.x;
    out = blockIdx.y;
    
    int h_in, w_in, h_out, w_out;
    h_out = threadIdx.x;
    w_out = threadIdx.y;
    h_in = h_out - 2;  //padding = 2
    w_in = w_out - 2;

    float sum = 0.;

    for(int in = 0; in < in_channels; in++){
        // load W to shared memory
        // just use h_out and w_out
        if(h_out < kernel_size && w_out < kernel_size){
            W_idx = out * in_channels * kernel_size * kernel_size + \
                    in * kernel_size * kernel_size + \
                    h_out * kernel_size + w_out;
            shared_W[h_out][w_out] = W[W_idx];
        }
        __syncthreads();


        // load X to shared memory
        if(h_in < feature_size) && (h_in >= 0) && (w_in < feature_size) && (w_in >=0)){
            X_idx = batch * 255 * 7 * 7 + \
                    in * 7 * 7 + \
                    h_out * 7 + w_out;
            shared_X[h_out][w_out] = X[X_idx];
        }else{
            shared_X[h_out][w_out] = 0;
        }
        __syncthreads();

        for (int p = 0; p < kernel_size; p++){
            for (int q = 0; q < kernel_size; q++){
                // have problem boundary check
                sum += shared_X[h_out + p][w_out + q] * shared_W[p][q];
            }
        }
        __syncthreads();
        
    }
    int Y_idx = batch *out_channels *feature_size *feature_size + \
            out * feature_size *feature_size + \
            h_out * feature_size + w_out;
    Y[Y_idx] = sum;
}

void launch_conv(float* Y,
                 const float* X,
                 const float* W,
                 int in_channels,
                 int out_channels,
                 int kernel_size,
                 int feature_size,
                 int batch_size){

    // for blocksize x,y for output size of feature map
    // since in this task, each feature map is small,
    // just set a const value?
    dim3 blockSize(7+4,7+4,1);
    // gridsize is for in channels and out channels
    dim3 gridSize(batch_size,out_channels,1);

}