__global__ void conv_kernel_exp(float *Y,
                                const float *X,
                                const float *W,
                                int in_channels,
                                int out_channels,
                                int kernel_size,
                                int feature_size,
                                int batch_size)
{
    // X: [1, 256, 7, 7]
    // batch x in_channel x kernel_size x kernel_size
    // W: [128, 256, 5, 5]
    //  out_channel x in_channels x kernel_size x kernel_size
    // Y: [1, 128, 7, 7]
    // batch x out_channels x kernel_size x kernel_size

    // int batch, out;

    // __shared__ float shared_X[7+2][7+2];
    // __shared__ float shared_W[5][5];
    // __shared__ float shared_Y[7][7];

    // batch = blockIdx.x;
    // out = blockIdx.y;

    // int h_in, w_in, h_out, w_out;
    // h_out = threadIdx.x;
    // w_out = threadIdx.y;
    // h_in = h_out - 2;  //padding = 2
    // w_in = w_out - 2;

    // float sum = 0.;
    // int X_idx,W_idx,Y_idx;
    // for(int in = 0; in < in_channels; in++){
    //     // load W to shared memory
    //     // just use h_out and w_out
    //     if(h_out < kernel_size && w_out < kernel_size){
    //         W_idx = out * in_channels * kernel_size * kernel_size + \
    //                 in * kernel_size * kernel_size + \
    //                 h_out * kernel_size + w_out;
    //         shared_W[h_out][w_out] = W[W_idx];
    //     }
    //     __syncthreads();
    //     // if(h_out < kernel_size && w_out < kernel_size){
    //     //     Y[h_out * kernel_size + w_out] = shared_W[h_out][w_out];
    //     // }

    //     // load X to shared memory
    //     if((h_out < feature_size) && (h_out >= 0) && (w_out < feature_size) && (w_out >=0)){
    //         X_idx = batch * 255 * 7 * 7 + \
    //                 in * 7 * 7 + \
    //                 h_out * 7 + w_out;
    //         shared_X[h_out][w_out] = X[X_idx];
    //     }else{
    //         shared_X[h_out][w_out] = 0;
    //     }
    //     __syncthreads();

    //     // if(h_out < feature_size && w_out < feature_size && \
    //     //     h_out + 2 > 0 && w_out + 2 > 0){
    //     //     Y[h_out * feature_size + w_out] = shared_X[h_out+2][w_out+2];
    //     // }

    //     if(h_out < feature_size && w_out < feature_size){
    //     for (int p = 0; p < kernel_size; p++){
    //         for (int q = 0; q < kernel_size; q++){
    //             // have problem boundary check
    //                 // sum += shared_X[h_out+p][w_out+q] * shared_W[p][q];
    //                 sum += shared_X[h_out+p][w_out+q] * shared_W[p][q];
    //             }
    //             // if(((h_in + p) >= 0) && ((h_in + p) < feature_size) && \
    //             //     ((w_in + q) >= 0) &&((w_in + q ) < feature_size) ){
    //             //     sum += shared_X[h_in + p][w_in + q] * shared_W[p][q];
    //             // }
    //         }
    //     }
    //     __syncthreads();

    // }
    // if(h_out < feature_size && w_out < feature_size){
    //     Y_idx = batch *out_channels *feature_size *feature_size + \
    //             out * feature_size *feature_size + \
    //             h_out * feature_size + w_out;
    //     Y[Y_idx] = sum;
    // }
    // }
}
__global__ void conv_kernel(float *Y,
                            const float *X,
                            const float *W,
                            int in_channels,
                            int out_channels,
                            int kernel_size,
                            int feature_size,
                            int batch_size)
{
    // X: [1, 256, 7, 7]
    // batch x in_channel x feature_size x feature_size
    // W: [128, 256, 5, 5]
    //  out_channel x in_channels x kernel_size x kernel_size
    // Y: [1, 128, 7, 7]
    // batch x out_channels x feature_size x feature_size

    int batch, out;

    __shared__ float shared_X[7][7];
    __shared__ float shared_W[5][5];
    // __shared__ float shared_Y[7][7];

    batch = blockIdx.x;
    out = blockIdx.y;

    // int h_in, w_in;
    int h_out, w_out;
    h_out = threadIdx.x;
    w_out = threadIdx.y;
    // h_in = h_out - 2;  //padding = 2
    // w_in = w_out - 2;

    float sum = 0.;

    int X_idx, W_idx, Y_idx;

    for (int in = 0; in < in_channels; in++)
    {
        // load W to shared memory
        // just use h_out and w_out
        if (h_out < kernel_size && w_out < kernel_size)
        {
            W_idx = out * in_channels * kernel_size * kernel_size +
                    in * kernel_size * kernel_size +
                    h_out * kernel_size + w_out;
            shared_W[h_out][w_out] = W[W_idx];
        }
        __syncthreads();

        // load X to shared memory
        if ((h_out < feature_size) && (h_out >= 0) && (w_out < feature_size) && (w_out >= 0))
        {
            X_idx = batch * 255 * 7 * 7 +
                    in * 7 * 7 +
                    h_out * 7 + w_out;
            shared_X[h_out][w_out] = X[X_idx];
        }
        __syncthreads();

        for (int p = 0; p < kernel_size; p++)
        {
            for (int q = 0; q < kernel_size; q++)
            {
                // have problem boundary check
                int h_idx = h_out - 2 + p;
                int w_idx = w_out - 2 + q;
                if (h_idx >= 0 && h_idx < feature_size &&
                    w_idx >= 0 && w_idx < feature_size)
                {
                    sum += shared_X[h_idx][w_idx] * shared_W[p][q];
                }
            }
        }
        __syncthreads();
    }
    Y_idx = batch * out_channels * feature_size * feature_size +
            out * feature_size * feature_size +
            h_out * feature_size + w_out;
    Y[Y_idx] = sum;
}

// for (int p = 0; p < kernel_size; p++){
//     for (int q = 0; q < kernel_size; q++){
//         // have problem boundary check
//         if(((h_out + p) >= 0) && ((h_out + p) < feature_size) && \
//             ((w_out + q) >= 0) &&((w_out +q ) < feature_size) ){
//             sum += shared_X[h_out + p][w_out + q] * shared_W[p][q];
//         }
//     }
// }

void launch_conv(float *Y,
                 const float *X,
                 const float *W,
                 int in_channels,
                 int out_channels,
                 int kernel_size,
                 int feature_size,
                 int batch_size)
{

    // for blocksize x,y for output size of feature map
    // since in this task, each feature map is small,
    // just set a const value?
    dim3 blockSize(7, 7, 1);
    // gridsize is for in channels and out channels
    dim3 gridSize(batch_size, out_channels, 1);

    conv_kernel<<<gridSize, blockSize>>>(Y,
                                         X,
                                         W,
                                         in_channels,
                                         out_channels,
                                         kernel_size,
                                         feature_size,
                                         batch_size);
}
__global__ void conv_kernel_exp(float *Y,
                                const float *X,
                                const float *W,
                                int in_channels,
                                int out_channels,
                                int kernel_size,
                                int feature_size,
                                int batch_size)
{
    // X: [1, 256, 7, 7]
    // batch x in_channel x kernel_size x kernel_size
    // W: [128, 256, 5, 5]
    //  out_channel x in_channels x kernel_size x kernel_size
    // Y: [1, 128, 7, 7]
    // batch x out_channels x kernel_size x kernel_size

    // int batch, out;

    // __shared__ float shared_X[7+2][7+2];
    // __shared__ float shared_W[5][5];
    // __shared__ float shared_Y[7][7];

    // batch = blockIdx.x;
    // out = blockIdx.y;

    // int h_in, w_in, h_out, w_out;
    // h_out = threadIdx.x;
    // w_out = threadIdx.y;
    // h_in = h_out - 2;  //padding = 2
    // w_in = w_out - 2;

    // float sum = 0.;
    // int X_idx,W_idx,Y_idx;
    // for(int in = 0; in < in_channels; in++){
    //     // load W to shared memory
    //     // just use h_out and w_out
    //     if(h_out < kernel_size && w_out < kernel_size){
    //         W_idx = out * in_channels * kernel_size * kernel_size + \
//                 in * kernel_size * kernel_size + \
//                 h_out * kernel_size + w_out;
    //         shared_W[h_out][w_out] = W[W_idx];
    //     }
    //     __syncthreads();
    //     // if(h_out < kernel_size && w_out < kernel_size){
    //     //     Y[h_out * kernel_size + w_out] = shared_W[h_out][w_out];
    //     // }

    //     // load X to shared memory
    //     if((h_out < feature_size) && (h_out >= 0) && (w_out < feature_size) && (w_out >=0)){
    //         X_idx = batch * 255 * 7 * 7 + \
//                 in * 7 * 7 + \
//                 h_out * 7 + w_out;
    //         shared_X[h_out][w_out] = X[X_idx];
    //     }else{
    //         shared_X[h_out][w_out] = 0;
    //     }
    //     __syncthreads();

    //     // if(h_out < feature_size && w_out < feature_size && \
//     //     h_out + 2 > 0 && w_out + 2 > 0){
    //     //     Y[h_out * feature_size + w_out] = shared_X[h_out+2][w_out+2];
    //     // }

    //     if(h_out < feature_size && w_out < feature_size){
    //     for (int p = 0; p < kernel_size; p++){
    //         for (int q = 0; q < kernel_size; q++){
    //             // have problem boundary check
    //                 // sum += shared_X[h_out+p][w_out+q] * shared_W[p][q];
    //                 sum += shared_X[h_out+p][w_out+q] * shared_W[p][q];
    //             }
    //             // if(((h_in + p) >= 0) && ((h_in + p) < feature_size) && \
//             //     ((w_in + q) >= 0) &&((w_in + q ) < feature_size) ){
    //             //     sum += shared_X[h_in + p][w_in + q] * shared_W[p][q];
    //             // }
    //         }
    //     }
    //     __syncthreads();

    // }
    // if(h_out < feature_size && w_out < feature_size){
    //     Y_idx = batch *out_channels *feature_size *feature_size + \
//             out * feature_size *feature_size + \
//             h_out * feature_size + w_out;
    //     Y[Y_idx] = sum;
    // }
    // }
}
__global__ void conv_kernel(float *Y,
                            const float *X,
                            const float *W,
                            int in_channels,
                            int out_channels,
                            int kernel_size,
                            int feature_size,
                            int batch_size)
{
    // X: [1, 256, 7, 7]
    // batch x in_channel x feature_size x feature_size
    // W: [128, 256, 5, 5]
    //  out_channel x in_channels x kernel_size x kernel_size
    // Y: [1, 128, 7, 7]
    // batch x out_channels x feature_size x feature_size

    int batch, out;

    __shared__ float shared_X[7][7];
    __shared__ float shared_W[5][5];
    // __shared__ float shared_Y[7][7];

    batch = blockIdx.x;
    out = blockIdx.y;

    // int h_in, w_in;
    int h_out, w_out;
    h_out = threadIdx.x;
    w_out = threadIdx.y;
    // h_in = h_out - 2;  //padding = 2
    // w_in = w_out - 2;

    float sum = 0.;

    int X_idx, W_idx, Y_idx;

    for (int in = 0; in < in_channels; in++)
    {
        // load W to shared memory
        // just use h_out and w_out
        if (h_out < kernel_size && w_out < kernel_size)
        {
            W_idx = out * in_channels * kernel_size * kernel_size +
                    in * kernel_size * kernel_size +
                    h_out * kernel_size + w_out;
            shared_W[h_out][w_out] = W[W_idx];
        }
        __syncthreads();

        // load X to shared memory
        if ((h_out < feature_size) && (h_out >= 0) && (w_out < feature_size) && (w_out >= 0))
        {
            X_idx = batch * 255 * 7 * 7 +
                    in * 7 * 7 +
                    h_out * 7 + w_out;
            shared_X[h_out][w_out] = X[X_idx];
        }
        __syncthreads();

        for (int p = 0; p < kernel_size; p++)
        {
            for (int q = 0; q < kernel_size; q++)
            {
                // have problem boundary check
                int h_idx = h_out - 2 + p;
                int w_idx = w_out - 2 + q;
                if (h_idx >= 0 && h_idx < feature_size &&
                    w_idx >= 0 && w_idx < feature_size)
                {
                    sum += shared_X[h_idx][w_idx] * shared_W[p][q];
                }
            }
        }
        __syncthreads();
    }
    Y_idx = batch * out_channels * feature_size * feature_size +
            out * feature_size * feature_size +
            h_out * feature_size + w_out;
    Y[Y_idx] = sum;
}

// for (int p = 0; p < kernel_size; p++){
//     for (int q = 0; q < kernel_size; q++){
//         // have problem boundary check
//         if(((h_out + p) >= 0) && ((h_out + p) < feature_size) && \
//             ((w_out + q) >= 0) &&((w_out +q ) < feature_size) ){
//             sum += shared_X[h_out + p][w_out + q] * shared_W[p][q];
//         }
//     }
// }

void launch_conv(float *Y,
                 const float *X,
                 const float *W,
                 int in_channels,
                 int out_channels,
                 int kernel_size,
                 int feature_size,
                 int batch_size)
{

    // for blocksize x,y for output size of feature map
    // since in this task, each feature map is small,
    // just set a const value?
    dim3 blockSize(7, 7, 1);
    // gridsize is for in channels and out channels
    dim3 gridSize(batch_size, out_channels, 1);

    conv_kernel<<<gridSize, blockSize>>>(Y,
                                         X,
                                         W,
                                         in_channels,
                                         out_channels,
                                         kernel_size,
                                         feature_size,
                                         batch_size);
}