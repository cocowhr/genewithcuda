#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
using namespace std;
#define NS 10
#define ST 10//T中的序列个数 T为参考模式库
#define CNUM 4//需要为2的倍数 现在分成4块 最多可以64
#define LEN 5
#define crossCNUM CNUM*(CNUM-1)
#define randomSize 500
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  
cudaError_t crossoverWithCuda(int *c[], int *a[]);
/*不用结构体 直接拆开用*/
__global__ void crossover(int **c,int **a,int* count,int *random)
{
	int x=CNUM/2*blockIdx.x+threadIdx.x;
	int y=CNUM/2*blockIdx.y+threadIdx.y;
	if(x>y)//x为j,y为i
	{	
		//printf("(%d,%d)\n",y,x); 
		//printf("(%d,%d)\n(%d)\n(%d)\n",y,x,a[y][0],a[x][0]); 
		int insert_pt=atomicAdd(count, 1);	
		// printf("%d\n",random[insert_pt]);
		 for(int i=0;i<random[insert_pt];i++)
		{
			// printf("%d\n",2*insert_pt);
			c[2*insert_pt][i]=a[y][i];
			c[2*insert_pt+1][i]=a[x][i];
		}
		for(int i=random[insert_pt];i<LEN;i++)
		{
			c[2*insert_pt][i]=a[x][i];
			c[2*insert_pt+1][i]=a[y][i];
		}
	}
}
int main()
{
	int* a[CNUM] = {};
	int* c[crossCNUM] = {};
	for (int i = 0; i < CNUM; i++)
	{
		a[i] = (int *)malloc(LEN*sizeof(int));
	}
	for (int i = 0; i < crossCNUM; i++)
	{
		c[i] = (int *)malloc(LEN*sizeof(int));
	}
	for (int i=0;i<CNUM;i++)
	{
		for(int j=0;j<LEN;j++)
		{
			a[i][j]=i+j+1;
		}
	}
	// Add vectors in parallel.
	cudaError_t cudaStatus = crossoverWithCuda(c, a);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	for (int i=0;i<crossCNUM;i++)
	{
		for(int j=0;j<LEN;j++)
		{
			cout<<c[i][j]<<" ";
		}
		cout<<endl;
	}
	// cudaThreadExit must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaThreadExit();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaThreadExit failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t crossoverWithCuda(int *c[], int *a[])
{
	//srand(time(0));
	int *random=new int[randomSize];
	for(int i=0;i<randomSize;i++)
	{
		random[i]=rand()%LEN;    
	}
	int *array_dev_input[CNUM] = {};
	int *array_dev_output[crossCNUM] = {};
	int *dev_rand=0; 
	int **dev_input = 0;
	int **dev_output = 0;
	size_t rows = CNUM;
	size_t cols = LEN;  
	CHECK(cudaMalloc((void**)&(array_dev_input), rows * sizeof(int*)));
	for (int i = 0; i < rows; i++)
	{
		CHECK(cudaMalloc((void**)&(array_dev_input[i]), cols * sizeof(int)));
		CHECK(cudaMemcpy(array_dev_input[i], a[i], cols * sizeof(int), cudaMemcpyHostToDevice));
	}
	for (int i = 0; i < crossCNUM; i++)
	{
		CHECK(cudaMalloc((void**)&(array_dev_output[i]), cols * sizeof(int)));
	}
	// Choose which GPU to run on, change this on a multi-GPU system.
	CHECK(cudaSetDevice(0));
	CHECK(cudaMalloc((void**)&(dev_input), rows * sizeof(int*)));
	CHECK(cudaMalloc((void**)&(dev_output), crossCNUM * sizeof(int*)));
	CHECK(cudaMemcpy(dev_input, array_dev_input, rows * sizeof(int*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_output, array_dev_output, crossCNUM * sizeof(int*), cudaMemcpyHostToDevice));
	// Allocate GPU buffers for three vectors (two input, one output)    

	// cudaThreadSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CHECK(cudaMalloc((void**)&(dev_rand), randomSize * sizeof(int)));
	// Copy input vectors from host memory to GPU buffers.
	int count=0;
	int *dev_count;
	CHECK(cudaMalloc((void**)&dev_count, sizeof(int)));
	CHECK(cudaMemcpy(dev_count, &count, sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_rand, random, randomSize * sizeof(int), cudaMemcpyHostToDevice));
	// Launch a kernel on the GPU with one thread for each element.
	dim3 threads(rows/2,rows/2);
	dim3 blocks(2,2);
	crossover<<<blocks, threads>>>(dev_output,dev_input,dev_count,dev_rand);
	// cudaThreadSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CHECK(cudaThreadSynchronize());
	for (int i = 0; i < crossCNUM; i++)
	{
		CHECK( cudaMemcpy(c[i], array_dev_output[i], cols * sizeof(int), cudaMemcpyDeviceToHost));
	}
	CHECK(cudaMemcpy(&count, dev_count, sizeof(int), cudaMemcpyDeviceToHost));
}