#include <iostream>
#include <math.h>

// This file deals with Pinned Memory and Pageable Memory. Will use nvprof to see how data transfer rates are affected

/* 
General Guideline:
1. Minimize the amount of data transferred between host and device when possible
2. Higher bandwidth is possible between the host and the device when using page-locked (or “pinned”) memory.
3. Batching many small transfers into one larger transfer performs much better because it eliminates most of the per-transfer overhead.

Host (CPU) data allocations are pageable by default. The GPU cannot access data directly from pageable host memory, 
so when a data transfer from pageable host memory to device memory is invoked, 
the CUDA driver must first allocate a temporary page-locked, or “pinned”, host array, 
followed by copying the host data to the pinned array, and then transfer the data from the pinned array to device memory

We can avoid the cost of the transfer between pageable and pinned host arrays by directly allocating our host arrays in pinned memory. 

Allocate pinned host memory in CUDA C/C++ using cudaMallocHost() or cudaHostAlloc(), and deallocate it with cudaFreeHost(). 
It is possible for pinned memory allocation to fail, so you should always check for errors

You should not over-allocate pinned memory. Doing so can reduce overall system performance because it reduces the amount of physical memory 
available to the operating system and other programs. How much is too much is difficult to tell in advance, so as with all optimizations,
test your applications and the systems they run on for optimal performance parameters.


*/

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
  
  // Pinned Memory directly instead of pageable memory
  // Can make a function to deal with the status instead of repeating the same code
  cudaError_t status = cudaMallocHost((void **)&x, N*sizeof(float));
  if (status != cudaSuccess)
    printf("Error allocating pinned host memory\n");
  status = cudaMallocHost((void **)&y, N*sizeof(float));
  if (status != cudaSuccess)
    printf("Error allocating pinned host memory\n");
  status = cudaMallocHost((void **)&z, N*sizeof(float));
  if (status != cudaSuccess)
    printf("Error allocating pinned host memory\n");

  float *d_x, *d_y;

  cudaMallocManaged(&d_x, N*sizeof(float));
  cudaMallocManaged(&d_y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  add<<<4, blockSize>>>(N, d_x, d_y);
  
  cudaMemcpy(z, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(z[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory, but use cudaFreeHost() instead of cudaFree()
  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(z);

  cudaFree(d_x);
  cudaFree(d_y);
  
  return 0;
}