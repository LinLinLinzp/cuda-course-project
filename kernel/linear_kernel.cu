#define TILE_WIDTH 16

__global__ void linear_kernel(float* Y,
                            const float* input_x,
                            const float* input_w,
                            int dim_xw,
                            int dim_xh,
                            int dim_ww,
                            int dim_wh
                            ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ float x_s[16];

    // load
    if(tid < dim_xw){
        x_s[tid] = input_x[tid];
    }
    __syncthreads();

    float sum = 0.0;
    // cal sum
    if(idx < dim_wh){
        for(int i = 0; i < dim_ww; i++){
            // sum += x_s[i] * input_w[idx][i];
            sum += x_s[i] * input_w[idx * dim_ww + i];
        }
    }
    // output
    if(idx < dim_wh){
        Y[idx] = sum;
    }
}

// __global__ void linear_kernel_2(float* Y,
//                             const float* input_x,
//                             const float* input_w,
//                             int dim_xw,
//                             int dim_xh,
//                             int dim_ww,
//                             int dim_wh
//                             ){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int tid = threadIdx.x;
//     int batch = blockIdx.y* blockDim.y + threadIdx.y;

//     __shared__ float x_s[16];

//     // load
//     if(idx < dim_xw){
//         x_s[tid] = input_x[idx + batch * dim_xw];
//     }
//     __syncthreads();

//     float sum = 0.0;
//     // cal sum
//     if(idx < dim_wh && batch < dim_xh){
//         for(int i = 0; i < dim_ww; i++){
//             // sum += x_s[i] * input_w[idx][i];
//             sum += x_s[i] * input_w[idx * dim_ww + i];
//         }
//     }
//     // output
//     if(idx < dim_wh && batch < dim_xh){
//         Y[idx + batch * dim_wh] = sum;
//     }
// }

// __global__ void linear_kernel_1(float* Y,
//                             const float* input_x,
//                             const float* input_w,
//                             int dim_xw,
//                             int dim_xh,
//                             int dim_ww,
//                             int dim_wh
//                             ){
//     __shared__ float shared_X[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float shared_W[TILE_WIDTH][TILE_WIDTH];

//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int Row = by*TILE_WIDTH + ty;
//     int Col = bx*TILE_WIDTH + tx;
//     float Pvalue = 0.0; //register

//     for(int m = 0; m < int(ceil((float)dim_wh / TILE_WIDTH)); ++m)
//     {
//         // load X to shared memory
//         if (m * TILE_WIDTH + tx < dim_xw && Row < dim_xh)
//         {
//             shared_X[ty][tx] = input_x[Row * dim_xw + m * TILE_WIDTH + tx];
//         }else{
//             //zero padding
//             shared_X[ty][tx] = 0.0;
//         }

//         //Load W to shared memory
//         if (m*TILE_WIDTH + ty < dim_wh && Col < dim_ww)
//         {
//             shared_W[ty][tx] = input_w[(m*TILE_WIDTH + ty)*dim_ww + Col];
//         }else{
//             //zero padding
//             shared_W[ty][tx] = 0.0;
//         }
//         __syncthreads();

//         for(int i = 0; i <TILE_WIDTH; ++i)
//         {
//             Pvalue += shared_X[ty][i] * shared_W[i][tx];
//         }
//         __syncthreads();

//         // write value to P
//         // if(Row < P.height && Col < P.width)
//         // {
//         //     Y[Row * P.width + Col] = Pvalue;
//         // }
//     }
    
// }



void launch_linear(float* device_y,
                    const float* input_x,
                    const float* input_w,
                    int input_dim_xw,
                    int input_dim_xh,
                    int input_dim_ww,
                    int input_dim_wh
                ){
    
    
    dim3 gridSize((input_dim_wh+1023)/1024,input_dim_xh);
    dim3 blockSize(1024,1);
    
    int dimx = (int)(ceil)((float)input_dim_ww / TILE_WIDTH);
    int dimy = (int)(ceil)((float)input_dim_xh / TILE_WIDTH);

    dim3 gridSize(dimx,dimy,1);
    dim3 blockSize(TILE_WIDTH,TILE_WIDTH,1);
    
    

    linear_kernel<<<gridSize, blockSize>>>(device_y, \
                                    input_x, \
                                    input_w, \
                                    input_dim_xw, \
                                    input_dim_xh, \
                                    input_dim_ww, \
                                    input_dim_wh
                                );
    
    
}