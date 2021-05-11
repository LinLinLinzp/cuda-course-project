
__global__ void convtranspose_kernel_1(float *Y,
                                        const float *X,
                                        const float *W,
                                        int in_channels,
                                        int out_channels,
                                        int kernel_size,
                                        int feature_size,
                                        int batch_size){

    // X: [1, 128, 7, 7]
    // batch x in_channel x feature_size x feature_size
    // W: [128, 64, 4, 4]
    //  in_channel x out_channels x kernel_size x kernel_size
    // Y: [1, 64, 14, 14]
    // batch x out_channels x feature_size x feature_size

    __shared__ float shared_X[31][31];
    __shared__ float shared_W[4][4];

    int batch, out ;
    batch = blockIdx.x;
    out = blockIdx.y;

    int h_out, w_out;
    h_out = threadIdx.x;
    w_out = threadIdx.y;

    float sum = 0.;

    int X_idx, W_idx, Y_idx;

    for (int in = 0; in < in_channels; in++){
        // load W to shared memory
        // no like conv, it's inverse load
        if (h_out < kernel_size && w_out < kernel_size){
            W_idx = in * out_channels * kernel_size * kernel_size +
                    out * kernel_size * kernel_size +
                    h_out * kernel_size + w_out;
            shared_W[kernel_size - h_out -1][kernel_size - w_out -1] = W[W_idx];
        }
        __syncthreads();
        
        //zero init of shared X
        shared_X[h_out][w_out] =0;

        // load X to shared memory
        // extend mapping
        if (h_out < feature_size && w_out < feature_size){
            X_idx = batch * in_channels * feature_size *feature_size + \
                    in * feature_size * feature_size + \
                    h_out * feature_size + w_out;
            shared_X[2 * h_out + 1][2 *w_out + 1] = X[X_idx];

        }
        __syncthreads();

        for (int p = 0; p < kernel_size; p++)
            {
                for (int q = 0; q < kernel_size; q++)
                {
                    // have problem boundary check
                    int h_idx = h_out - 1 + p;
                    int w_idx = w_out - 1 + q;
                    if (h_idx >= 0 && h_idx < feature_size * 2 &&
                        w_idx >= 0 && w_idx < feature_size * 2)
                    {
                        sum += shared_X[h_idx][w_idx] * shared_W[p][q];
                    }
                }
            }
        __syncthreads();
    }
    Y_idx = batch * out_channels * feature_size * feature_size * 4 +
            out * feature_size * feature_size * 4 +
            h_out * feature_size * 2 + w_out;
    Y[Y_idx] = sum;
}


void launch_convtranspose_1(float *Y,
                            const float *X,
                            const float *W,
                            int in_channels,
                            int out_channels,
                            int kernel_size,
                            int feature_size,
                            int batch_size,
                            int stride){

    int output_size = feature_size * stride;
    //each threads correspond to one element 
    dim3 blockSize(output_size,output_size,1);

    dim3 gridSize(batch_size,out_channels,1);
                            
    convtranspose_kernel_1<<<gridSize, blockSize>>>(Y,
                                                    X,
                                                    W,
                                                    in_channels,
                                                    out_channels,
                                                    kernel_size,
                                                    feature_size,
                                                    batch_size
                                                    );

}

void launch_convtranspose_2(float *Y,
                            const float *X,
                            const float *W,
                            int in_channels,
                            int out_channels,
                            int kernel_size,
                            int feature_size,
                            int batch_size){



}

