#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float *y)
{
 
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}


int main(void)
{
  int N = 1<<20;
  float *x, *y, *z;
  
  x = (float *)malloc(N*sizeof(float));
  y = (float *)malloc(N*sizeof(float));
  z = (float *)malloc(N*sizeof(float));

  float *d_x, *d_y;

  // cudaMalloc((void **)&d_x, N*sizeof(float));
  // cudaMalloc((void **)&d_y, N*sizeof(float));
  // cudaMalloc((void **)&d_z, N*sizeof(float));

  // Allocate Unified Memory – accessible from CPU or GPU
  // The data is automatically migrated between the CPU and GPU when necessary, without explicit data transfer. 
  // This can simplify the programming model, as the developer does not need to manage data transfers explicitly.
  cudaMallocManaged(&d_x, N*sizeof(float));
  cudaMallocManaged(&d_y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  //   CUDA kernel launches are specified using the triple angle bracket syntax <<< >>>
  add<<<4, blockSize>>>(N, d_x, d_y);
  
  cudaMemcpy(z, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(z[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  free(x);
  free(y);
  free(z);

  cudaFree(d_x);
  cudaFree(d_y);
  
  return 0;
}