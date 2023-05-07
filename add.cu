#include <iostream>
#include <math.h>
//  Kernel function to add the elements of two arrays
//  the specifier __global__ is added to the function, which tells the CUDA C++ compiler that 
//  this is a function that runs on the GPU and can be called from CPU code.
// __global__
// void add(int n, float *x, float *y)
// {
//   for (int i = 0; i < n; i++)
//     y[i] = x[i] + y[i];
// }


// Now this code below is when we have one block and multiple threads for that block <<<1, 256>>>
// __global__
// void add(int n, float *x, float *y)
// {
    // threadIdx.x contains the index of the current thread within its block,
    // and blockDim.x contains the number of threads in the block. 
//   int index = threadIdx.x;
//   int stride = blockDim.x;
    // modify the loop to stride through the array with parallel threads.
//   for (int i = index; i < n; i += stride)
//       y[i] = x[i] + y[i];
// }

__global__
void add(int n, float *x, float *y)
{
  // CUDA provides gridDim.x, which contains the number of blocks in the grid
  // and blockIdx.x, which contains the index of the current thread block in the grid.
  // Since the grid is represented as a one-dimensional array
  // we can use these two variables to uniquely identify each thread in the grid.
  // The idea is that each thread gets its index by computing the offset to the beginning of its block 
  // (the block index times the block size: blockIdx.x * blockDim.x) and adding the thread’s index within the block (threadIdx.x)
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}


int main(void)
{
  int N = 1<<20;
  float *x, *y;
  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU

  // let’s start by changing the second one: The number of threads in a thread block. <<<1, X>>>
  // CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose.
  
  // If I run the code with only this change, it will do the computation once per thread, 
  // rather than spreading the computation across the parallel threads
  // add<<<1, 256>>>(N, x, y);

  //  CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors, or SMs. Each SM can run multiple concurrent thread blocks
  //  the first parameter of the execution configuration specifies the number of thread block
  //  the blocks of parallel threads make up what is known as the grid.
  //  
  // Since I have N elements to process, and 256 threads per block, I just need to calculate the number of blocks to get at least N threads. 
  // I simply divide N by the block size (being careful to round up in case N is not a multiple of blockSize).
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;



  //   CUDA kernel launches are specified using the triple angle bracket syntax <<< >>>
  add<<<4, blockSize>>>(N, x, y);
  

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}