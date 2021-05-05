__global__ void conv_kernel(float* Y,
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