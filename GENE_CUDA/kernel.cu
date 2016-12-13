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
#define ST 5//T中的序列个数 T为参考模式库
#define CNUM 4//需要为2的倍数 现在分成4块 最多可以64
#define LEN 5
#define crossCNUM CNUM*(CNUM-1)
#define randomSize 500
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  
cudaError_t geneWithCuda(int *crossseq[], int *seq[],int *ref[],double *fit);
/*不用结构体 直接拆开用*/
__device__ int randomnum;
//__device__  void cross_calculate_fit(double ci1,double ci2,int *seq1,int *seq2,int **ref,double &fit1,double&fit2)
//{
//	int M1=0;
//	int M2=0;
//	for(int k=0;k<ST;k++)
//	{
//		bool eq1=true;
//		bool eq2=true;
//		for(int m=0;m<LEN;m++)
//		{
//			int refkm=ref[k][m];
//			if(eq1)
//			{
//				if(0!=seq1[m]&&refkm!=seq1[m])
//				{
//					eq1=false;
//				}
//			}
//			if(eq2)
//			{
//				if(0!=seq2[m]&&refkm!=seq2[m])
//				{
//					eq2=false;
//				}
//			}
//			if(!eq1&&!eq2)
//			{
//				break;
//			}
//		}
//		if(eq1)
//		{
//			M1++;
//		}
//		if(eq2)
//		{
//			M2++;
//		}
//	}	
//	fit1=ci1*M1*_Pow_int(NS,LEN)/ST;
//	fit2=ci2*M2*_Pow_int(NS,LEN)/ST;
//}
__device__  void calculate_fit(double ci,int *seq,int **ref,double &fit)
{
	int M=0;
	for(int k=0;k<ST;k++)
	{
		bool eq=true;
		for(int m=0;m<LEN;m++)
		{
			int refkm=ref[k][m];
			if(eq)
			{
				if(0!=seq[m]&&refkm!=seq[m])
				{
					eq=false;
					break;
				}
			}
		}
		if(eq)
		{
			M++;
		}
	}	
	fit=ci*M*_Pow_int(NS,LEN)/ST;
}
__global__ void crossover(int **crossseq,int **seq,int **ref,int* crosscount,int *random)
{

	int x=CNUM/2*blockIdx.x+threadIdx.x;
	int y=CNUM/2*blockIdx.y+threadIdx.y;
	if(x>y)//x为j,y为i
	{	
		//printf("(%d,%d)\n",y,x); 
		//printf("(%d,%d)\n(%d)\n(%d)\n",y,x,seq[y][0],seq[x][0]); 
		int insert_pt=atomicAdd(crosscount, 2);
		int iscross=atomicAdd(&randomnum, 1);
		// printf("%d\n",random[insert_pt]);
		if(random[iscross]<80)
		{
			int crossloc=atomicAdd(&randomnum, 1);
			for(int i=0;i<random[crossloc]%LEN;i++)
			{
				// printf("%d\n",2*insert_pt);
				crossseq[insert_pt][i]=seq[y][i];
				crossseq[insert_pt+1][i]=seq[x][i];
			}
			for(int i=random[crossloc]%LEN;i<LEN;i++)
			{
				crossseq[insert_pt][i]=seq[x][i];
				crossseq[insert_pt+1][i]=seq[y][i];
			}
			//cross_calculate_fit(ci1,ci2,crossseq[insert_pt],crossseq[insert_pt+1],ref,fit[insert_pt],fit[insert_pt+1]);
		}
	}

}
__global__ void mutation(int **mutationseq,int **crossseq,int **ref,double*fit,int *random)
{
	int x=CNUM/2*blockIdx.x+threadIdx.x;
	double ci=0.5;
	//printf("(%d)\n",x); 
	//printf("(%d,%d)\n(%d)\n(%d)\n",y,x,a[y][0],a[x][0]); 
	// printf("%d\n",random[insert_pt]);
	for(int i=0;i<LEN;i++)
	{
		int ismuta=atomicAdd(&randomnum ,1);
		if(random[ismuta]<5)//random<5的概率为5%;
		{		
			int mutanum=atomicAdd(&randomnum ,1);
			mutationseq[x][i]=random[mutanum]%NS+1;
		}
		else
		{
			mutationseq[x][i]=crossseq[x][i];
		}
	}	
	calculate_fit(ci,mutationseq[x],ref,fit[x]);
}
int main()
{
	int* seq[CNUM] = {};
	int* crossseq[crossCNUM] = {};
	int *ref[ST]={};
	double *fit=new double[crossCNUM];
	for (int i = 0; i < CNUM; i++)
	{
		seq[i] =new int[LEN];
	}
	for (int i = 0; i < crossCNUM; i++)
	{
		crossseq[i]=new int[LEN];
	}
	for(int i=0;i<ST;i++)
	{
		ref[i]=new int [LEN];
	}
	seq[0][0]=1;
	seq[0][1]=2;
	seq[0][2]=3;
	seq[0][3]=4;
	seq[0][4]=5;
	//12345
	seq[1][0]=4;
	seq[1][1]=2;
	seq[1][2]=8;
	seq[1][3]=6;
	seq[1][4]=3;
	//42863
	seq[2][0]=5;
	seq[2][1]=2;
	seq[2][2]=1;
	seq[2][3]=4;
	seq[2][4]=7;
	//52147
	seq[3][0]=3;
	seq[3][1]=3;
	seq[3][2]=2;
	seq[3][3]=5;
	seq[3][4]=6;
	//33256


	ref[0][0]=1;
	ref[0][1]=2;
	ref[0][2]=3;
	ref[0][3]=4;
	ref[0][4]=7;
	//12347
	ref[1][0]=1;
	ref[1][1]=2;
	ref[1][2]=3;
	ref[1][3]=4;
	ref[1][4]=7;
	//12347
	ref[2][0]=5;
	ref[2][1]=2;
	ref[2][2]=3;
	ref[2][3]=4;
	ref[2][4]=5;
	//52345
	ref[3][0]=3;
	ref[3][1]=3;
	ref[3][2]=2;
	ref[3][3]=8;
	ref[3][4]=6;
	//33286
	ref[4][0]=1;
	ref[4][1]=2;
	ref[4][2]=2;
	ref[4][3]=5;
	ref[4][4]=6;
	//12256
	// Add vectors in parallel.
	cudaError_t cudaStatus = geneWithCuda(crossseq, seq,ref,fit);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	for (int i=0;i<crossCNUM;i++)
	{
		for(int j=0;j<LEN;j++)
		{
			cout<<crossseq[i][j]<<" ";
		}
		cout<<endl;
		cout<<fit[i]<<endl;
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
cudaError_t geneWithCuda(int *outputseq[], int *seq[],int *ref[],double *fit)
{
	//srand(time(0));
	int *random=new int[randomSize];
	for(int i=0;i<randomSize;i++)
	{
		random[i]=rand()%100;    
	}
	int crosscount=0;//TODO:记得改成0
	int *array_dev_input[CNUM] = {};
	int *array_dev_ref[ST]={};
	int *array_dev_crossoutput[crossCNUM] = {};
	int *array_dev_mutaoutput[crossCNUM] = {};
	int *dev_rand=0;
	double *dev_fit=0;
	int **dev_input = 0;
	int **dev_ref=0;
	int **dev_crossoutput = 0;
	int **dev_mutaoutput = 0;
	int *dev_crosscount;

	CHECK(cudaSetDevice(0));
	CHECK(cudaMalloc((void**)&(array_dev_input), CNUM * sizeof(int*)));
	for (int i = 0; i < CNUM; i++)
	{
		CHECK(cudaMalloc((void**)&(array_dev_input[i]), LEN * sizeof(int)));
		CHECK(cudaMemcpy(array_dev_input[i], seq[i], LEN * sizeof(int), cudaMemcpyHostToDevice));
	}
	for (int i = 0; i < crossCNUM; i++)
	{
		CHECK(cudaMalloc((void**)&(array_dev_crossoutput[i]), LEN * sizeof(int)));
		CHECK(cudaMalloc((void**)&(array_dev_mutaoutput[i]), LEN * sizeof(int)));
	}
	for (int i = 0; i < ST; i++)
	{
		CHECK(cudaMalloc((void**)&(array_dev_ref[i]), LEN * sizeof(int)));
		CHECK(cudaMemcpy(array_dev_ref[i], ref[i], LEN * sizeof(int), cudaMemcpyHostToDevice));
	}
	CHECK(cudaMalloc((void**)&(dev_input), CNUM * sizeof(int*)));
	CHECK(cudaMalloc((void**)&(dev_crossoutput), crossCNUM * sizeof(int*)));
	CHECK(cudaMalloc((void**)&(dev_mutaoutput), crossCNUM * sizeof(int*)));
	CHECK(cudaMalloc((void**)&(dev_ref), ST * sizeof(int*)));
	CHECK(cudaMalloc((void**)&(dev_rand), randomSize * sizeof(int)));
	CHECK(cudaMalloc((void**)&(dev_fit), crossCNUM * sizeof(double)));
	CHECK(cudaMalloc((void**)&(dev_crosscount), sizeof(int)));

	CHECK(cudaMemcpy(dev_input, array_dev_input, CNUM * sizeof(int*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_crossoutput, array_dev_crossoutput, crossCNUM * sizeof(int*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_mutaoutput, array_dev_mutaoutput, crossCNUM * sizeof(int*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_ref, array_dev_ref, ST * sizeof(int*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_crosscount, &crosscount, sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_rand, random, randomSize * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_fit, fit, crossCNUM * sizeof(double), cudaMemcpyHostToDevice));


	dim3 threads(CNUM/2,CNUM/2);
	dim3 blocks(2,2);
	crossover<<<blocks, threads>>>(dev_crossoutput,dev_input,dev_ref,dev_crosscount,dev_rand);
	CHECK(cudaThreadSynchronize());
	CHECK(cudaMemcpy(&crosscount, dev_crosscount, sizeof(int), cudaMemcpyDeviceToHost));
	mutation<<<1,crosscount>>>(dev_mutaoutput,dev_crossoutput,dev_ref,dev_fit,dev_rand);
	CHECK(cudaThreadSynchronize());
	for (int i = 0; i < crossCNUM; i++)
	{
		CHECK( cudaMemcpy(outputseq[i], array_dev_crossoutput[i], LEN * sizeof(int), cudaMemcpyDeviceToHost));
	}
	
	//for (int i = 0; i < crossCNUM; i++)
	//{
	//	CHECK( cudaMemcpy(outputseq[i], array_dev_mutaoutput[i], LEN * sizeof(int), cudaMemcpyDeviceToHost));
	//}
	CHECK(cudaMemcpy(fit, dev_fit, crossCNUM*sizeof(double), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(&crosscount, dev_crosscount, sizeof(int), cudaMemcpyDeviceToHost));
}