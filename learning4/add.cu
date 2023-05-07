#include <iostream>
#include <math.h>

// This file deals with CUDA Streams

// A stream in CUDA is a sequence of operations that execute on the device in the order in which they are issued by the host code (FIFO)

// While operations within a stream are guaranteed to execute in the prescribed order,
// operations in different streams can be interleaved and, when possible, they can even run concurrently.

/* 
When no stream is specified, the default stream (also called the “null stream”) is used.
The default stream is different from other streams because it is a synchronizing stream with respect to operations on the device

no operation in the default stream will begin until all previously issued operations in any stream on the device have completed, 
and an operation in the default stream must complete before any other operation

The asynchronous behavior of kernel launches from the host’s perspective makes overlapping device and host computation very simple.

Non-default streams in CUDA C/C++ are declared, created, and destroyed in host code

To issue a data transfer to a non-default stream we use the cudaMemcpyAsync() function, 
which is similar to the cudaMemcpy() function but takes a stream identifier as a fifth argument.

To issue a kernel to a non-default stream we specify the stream identifier as a fourth execution configuration parameter
(the third execution configuration parameter allocates shared device memory, which we’ll talk about later; use 0 for now).

Since all operations in non-default streams are non-blocking with respect to the host code, 
you will run across situations where you need to synchronize the host code with operations in a stream. 


The “heavy hammer” way is to use cudaDeviceSynchronize(), which blocks the host code until all previously issued operations on the device have completed. 
In most cases this is overkill, and can really hurt performance due to stalling the entire device and host thread.

1. The function cudaStreamSynchronize(stream) can be used to block the host thread 
until all previously issued operations in the specified stream have completed

2. The function cudaStreamQuery(stream) tests whether all operations issued to the specified stream have completed, without blocking host execution

3. You can also synchronize operations within a single stream on a specific event using cudaStreamWaitEvent(event)

For overlapping kernel execution and data transfers:
1. The kernel execution and the data transfer to be overlapped must both occur in different, non-default streams.
2. The host memory involved in the data transfer must be pinned memory. (learning3)

In the modified code, we break up the array of size N into chunks of streamSize elements

Since the kernel operates independently on all elements, each of the chunks can be processed independently. 

The number of (non-default) streams used is nStreams=N/streamSize. 
There are multiple ways to implement the domain decomposition of the data and processing
one way is to loop over all the operations for each chunk of the array

Another approach is to batch similar operations together, issuing all the host-to-device transfers first, 
followed by all kernel launches, and then all device-to-host transfers

The two approaches perform very differently depending on the specific generation of GPU used.

Consider the C1060, which has a single copy engine and a single kernel engine
H2D = Host to Device, D2H = Device to Host, K = Kernel Launch
The organization of first approach can be represented as {(H2D-1, K-1, D2H-1), (H2D-2, K-2, D2H-2), (H2D-3, K-3, D2H-3)}

The organization of second approach can be represented as {(H2D-1, H2D-2, H2D-3), (K-1, K-2, K-3), (D2H-1, DSH-2, D2H-3)}

we do not see any speed-up when using the first asynchronous version on the C1060: tasks were issued to the copy engine in an order 
that precludes any overlap of kernel execution and data transfer. 
For version two, however, where all the host-to-device transfers are issued before any of the device-to-host transfers, 
overlap is possible as indicated by the lower execution time.


Now consider the C2050 has two copy engines, one for host-to-device transfers and another for device-to-host transfers, as well as a single kernel engine.

For approach 1, the device-to-host transfer of data in stream[i] does not block the host-to-device transfer of data in stream[i+1]
as it did on the C1060 because there is a separate engine for each copy direction on the C2050. 

the performance degradation observed in asynchronous version 2 on the C2050 is related to the it's ability to concurrently run multiple 
kernels. When multiple kernels are issued back-to-back in different (non-default) streams, the scheduler tries to enable concurrent 
execution of these kernels and as a result delays a signal that normally occurs after each kernel completion 
(which is responsible for kicking off the device-to-host transfer) until all kernels complete. So, while there is overlap between 
host-to-device transfers and kernel execution in the second version of our asynchronous code, there is no overlap between 
kernel execution and device-to-host transfers

The NVIDIA 1650 on my laptop has one copy engine and one kernel engine, so the first approach is the best for this GPU
*/

__global__
void add(int n, float *x, float *y){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}


int main(void){
	int N = 1<<20;
	float *x, *y, *z;
  
  // Pinned Memory directly instead of pageable memory. Non default streams need pinned memory
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
	int nStreams = 4;
	int streamSize = N / nStreams;

	cudaStream_t stream[nStreams];
	cudaError_t result;

	for(int i = 0; i < nStreams; i++){
		// The stream is created and stored in the array. It later needs to be destroyed
		result = cudaStreamCreate(&stream[i]);
		if (result != cudaSuccess)
			printf("Error creating stream\n");
	}
	// Approach 1. Suitable for GPUs with one copy engine and one kernel engine
	for(int i = 0; i < nStreams; i++) {
		// The point where each stream needs to start from
		int offset = i * streamSize;
		// We send the data to the device asynchronously and use the offset to determine where to start
		// cudaMemcpyAsync is used instead of cudaMemcpy for non-default streams
		cudaMemcpyAsync(&d_x[offset], &x[offset], streamSize*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
		cudaMemcpyAsync(&d_y[offset], &y[offset], streamSize*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
		// If we wanted to pass d_x and d_y, then we would have to pass the offset parameter as well, and manage the for loop in the kernel
		// Instead we can just pass the pointers to the correct location in the array
		add<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(streamSize, &d_x[offset], &d_y[offset]);
		cudaMemcpyAsync(&z[offset], &d_y[offset], streamSize*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
	}

	// Approach 2. Suitable for GPUs with multiple copy engines and one kernel engine
	// for(int i = 0; i < nStreams; i++) {
	// 	int offset = i * streamSize;
	// 	cudaMemcpyAsync(&d_x[offset], &x[offset], streamSize*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
	// 	cudaMemcpyAsync(&d_y[offset], &y[offset], streamSize*sizeof(float), cudaMemcpyHostToDevice, stream[i]);
	// }

	// for(int i = 0; i < nStreams; i++) {
	// 	int offset = i * streamSize;
	// 	add<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(streamSize, &d_x[offset], &d_y[offset]);
	// }

	// for(int i = 0; i < nStreams; i++) {
	// 	int offset = i * streamSize;
	// 	cudaMemcpyAsync(&z[offset], &d_y[offset], streamSize*sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
	// }

  // Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++){
		maxError = fmax(maxError, fabs(z[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	// Destroy all the streams
	for (int i = 0; i < nStreams; ++i){
		result = cudaStreamDestroy(stream[i]);
		if (result != cudaSuccess)
			printf("Error destroying stream\n");
	}

  // Free memory, but use cudaFreeHost() instead of cudaFree()
	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(z);

	cudaFree(d_x);
	cudaFree(d_y);
  
  return 0;
}