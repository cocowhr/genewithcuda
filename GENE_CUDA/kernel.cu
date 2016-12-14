#include <WinSock2.h>  
#include "mysql.h"  
#pragma comment(lib,"wsock32.lib")  
#pragma comment(lib,"libmysql.lib")  
#include <atlstr.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <algorithm>
#include <vector>
using namespace std;
#define NS 10
#define ST 100000//T中的序列个数 T为参考模式库 
#define CNUM 4//群体规模 需要为2的倍数 现在分成4块 最多可以64
#define NUM 5//迭代次数
#define LEN 7
#define crossCNUM CNUM*(CNUM-1)
#define randomSize 500
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}  
cudaError_t geneWithCuda(int *seq[],int *outputseq[], double *inputfit,double *outputfit,int& count);
class Chrom
{
public:
	Chrom() 
	{
		seq=new int[LEN];
		fit=0;
	}
	Chrom(const Chrom& a)
	{
		seq=new int[LEN];
		for(int i=0;i<LEN;i++)
		{
			seq[i]=a.seq[i];
		}
		fit=a.fit;
		//MM.assign(a.MM.begin(),a.MM.end());
	}
	~Chrom() {}
	int* seq ;
	double fit;//适应值
};
bool Comp(Chrom& first,Chrom& second)
{
	return first.fit > second.fit;
}
typedef struct Code                           // 结构体类型，为单个染色体的结构；
{
	int id;//0号预留给*
	double count;
}code;    
int *refer[ST]={};//因为数组太大 需要放在外面
__host__ void evpop (int **seq,int **ref)   // 函数：随机生成初始种群；//todo:需要GPU计算fit
{
	int random ;
	for(int i=0;i<CNUM;i++)
	{
		for(int j=0;j<LEN;j++)
		{
			random=rand ()%NS;                     // 产生一个随机值
			seq[i][j]=random;
		}
	}
	for(int i=0;i<CNUM;i++)
	{
		int random1=rand ()%3;
		for(int j=0;j<random1;j++)
		{
			random=rand ()%LEN; 
			seq[i][random]=0;
		}
	}       
}       
__device__ int randomnum;
__device__  void calculate_fit(double ci,int E,int *seq,int **devref,double &fit)//计算fit值
{
	int M=0;
	for(int k=0;k<ST;k++)
	{
		bool eq=true;
		for(int m=0;m<LEN;m++)
		{
			int refkm=devref[k][m];
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
	fit=ci*M*_Pow_int(NS,E)/ST;
}
__global__ void calculate_fit_input(int **seq,int ** devref,double *fit)
{
	int x=threadIdx.x;
	double ci=0;
	int E=LEN;
	for(int i=0;i<LEN;i++)
	{
		if(seq[x][i]!=0)
		{
			ci+=0.1;
		}
		else
		{
			E--;
		}
	}	
	calculate_fit(ci,E,seq[x],devref,fit[x]);
}
__global__ void crossover(int **crossseq,int **seq,int **devref,int* crosscount,int *random)//交叉操作
{

	int x=CNUM/2*blockIdx.x+threadIdx.x;
	int y=CNUM/2*blockIdx.y+threadIdx.y;
	if(x>y)//x为j,y为i
	{	
		//printf("(%d,%d)\n",y,x); 
		//printf("(%d,%d)\n(%d)\n(%d)\n",y,x,seq[y][0],seq[x][0]); 
		int iscross=atomicAdd(&randomnum, 1);
		// printf("%d\n",random[insert_pt]);
		if(random[iscross]<80)
		{
			int insert_pt=atomicAdd(crosscount, 2);
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
		}
	}

}
__global__ void mutation(int **mutationseq,int **crossseq,int **devref,double*fit,int *random) // 变异操作；
{
	int x=threadIdx.x;
	double ci=0;
	int E=LEN;
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
		if(mutationseq[x][i]!=0)
		{
			ci+=0.1;
		}
		else
		{
			E--;
		}
	}	
	calculate_fit(ci,E,mutationseq[x],devref,fit[x]);
}
int main()
{
	int* seq[CNUM] = {};
	MYSQL mysql;  
	MYSQL_RES *result;  
	MYSQL_ROW row;  
	mysql_init(&mysql);  
	mysql_real_connect(&mysql, "localhost", "root", "root", "genet", 3306, NULL, 0); 
	CString strSQL;
	strSQL.Format("SELECT * FROM seq");
	mysql_query(&mysql, strSQL);
	result = mysql_store_result(&mysql);  
	for (int i = 0; i < CNUM; i++)
	{
		seq[i] =new int[LEN];
	}
	for(int i=0;i<ST;i++)
	{
		refer[i]=new int [LEN];
	}
	for(int i=0;i<ST;i++)
	{  
		row = mysql_fetch_row(result);
		for(int j=0;j<LEN;j++)
		{
			refer[i][j]=atoi(row[j+1]);
		}
	}  
	evpop(seq,refer);
	//for(int i=0;i<CNUM;i++)
	//{
	//	for(int j=0;j<LEN;j++)
	//	{
	//		cout<<seq[i][j]<<" ";
	//	}
	//	cout<<endl;
	//}
	//cout<<"********************************"<<endl;
	// Add vectors in parallel.
	vector<Chrom>chroms;
	for(int t =0;t< NUM;t ++)                          // 开始迭代；
	{
		int* outputseq[crossCNUM] = {};
		double *inputfit=new double[CNUM];
		double *outputfit=new double[crossCNUM];
		for (int i = 0; i < crossCNUM; i++)
		{
			outputseq[i]=new int[LEN];
		}
		printf("\ni = %d\n" ,t);                 // 输出当前迭代次数；
		int count=0;
		cudaError_t cudaStatus = geneWithCuda( seq,outputseq,inputfit,outputfit,count);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}
		cudaStatus = cudaThreadExit();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaThreadExit failed!");
			return 1;
		}
		chroms.clear();
		for(int i=0;i<CNUM;i++)
		{
			Chrom temp;
			for(int j=0;j<LEN;j++)
			{
				temp.seq[j]=seq[i][j];
			}
			temp.fit=inputfit[i];
			chroms.push_back(temp);
		}
		for(int i=0;i<count;i++)
		{
			Chrom temp;
			for(int j=0;j<LEN;j++)
			{
				temp.seq[j]=outputseq[i][j];
			}
			temp.fit=outputfit[i];
			chroms.push_back(temp);
		}
		sort(chroms.begin(),chroms.end(),Comp);
		//for(int i=0;i<chroms.size();i++)
		//{
		//	for(int j=0;j<LEN;j++)
		//	{
		//		cout<<chroms[i].seq[j]<<" ";
		//	}
		//	cout<<endl;
		//	cout<<chroms[i].fit<<endl;
		//}
		for(int i=0;i<CNUM;i++)
		{
			for(int j=0;j<LEN;j++)
			{
				seq[i][j]=chroms[i].seq[j];
			}
		}
		delete[] outputfit;
		delete[] inputfit;
		for(int i=0;i<crossCNUM;i++)
		{
			delete  outputseq[i];
		}	
		// cudaThreadExit must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		
	}
		for(int i=0;i<CNUM;i++)
		{
			for(int j=0;j<LEN;j++)
			{
				cout<<chroms[i].seq[j]<<" ";
			}
			cout<<endl;
			cout<<chroms[i].fit<<endl;
		}
	return 0;
}
// Helper function for using CUDA to add vectors in parallel.
/*
交叉变异操作交给GPU 
输入参数为：
seq 父种群
ref 参考模式库
输出参数为:
outputseq 子种群
inputfit:父种群的适应值
outputfit:子种群的适应值
count:子种群数量
*/
cudaError_t geneWithCuda(int *seq[],int *outputseq[], double*inputfit,double *outputfit,int &count)
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
	double *dev_inputfit=0;
	double *dev_outputfit=0;
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
		CHECK(cudaMemcpy(array_dev_ref[i], refer[i], LEN * sizeof(int), cudaMemcpyHostToDevice));
	}
	CHECK(cudaMalloc((void**)&(dev_input), CNUM * sizeof(int*)));
	CHECK(cudaMalloc((void**)&(dev_crossoutput), crossCNUM * sizeof(int*)));
	CHECK(cudaMalloc((void**)&(dev_mutaoutput), crossCNUM * sizeof(int*)));
	CHECK(cudaMalloc((void**)&(dev_ref), ST * sizeof(int*)));
	CHECK(cudaMalloc((void**)&(dev_rand), randomSize * sizeof(int)));
	CHECK(cudaMalloc((void**)&(dev_inputfit), CNUM * sizeof(double)));
	CHECK(cudaMalloc((void**)&(dev_outputfit), crossCNUM * sizeof(double)));
	CHECK(cudaMalloc((void**)&(dev_crosscount), sizeof(int)));

	CHECK(cudaMemcpy(dev_input, array_dev_input, CNUM * sizeof(int*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_crossoutput, array_dev_crossoutput, crossCNUM * sizeof(int*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_mutaoutput, array_dev_mutaoutput, crossCNUM * sizeof(int*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_ref, array_dev_ref, ST * sizeof(int*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_crosscount, &crosscount, sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_rand, random, randomSize * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_inputfit, inputfit, CNUM * sizeof(double), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_outputfit, outputfit, crossCNUM * sizeof(double), cudaMemcpyHostToDevice));


	calculate_fit_input<<<1,CNUM>>>(dev_input,dev_ref,dev_inputfit);
	CHECK(cudaThreadSynchronize());
	CHECK(cudaMemcpy(inputfit, dev_inputfit, CNUM*sizeof(double), cudaMemcpyDeviceToHost));
	dim3 threads(CNUM/2,CNUM/2);
	dim3 blocks(2,2);
	crossover<<<blocks, threads>>>(dev_crossoutput,dev_input,dev_ref,dev_crosscount,dev_rand);
	CHECK(cudaThreadSynchronize());
	CHECK(cudaMemcpy(&crosscount, dev_crosscount, sizeof(int), cudaMemcpyDeviceToHost));
	mutation<<<1,crosscount>>>(dev_mutaoutput,dev_crossoutput,dev_ref,dev_outputfit,dev_rand);
	CHECK(cudaThreadSynchronize());
	//for (int i = 0; i < crossCNUM; i++)
	//{
	//	CHECK( cudaMemcpy(outputseq[i], array_dev_crossoutput[i], LEN * sizeof(int), cudaMemcpyDeviceToHost));
	//}

	for (int i = 0; i < crossCNUM; i++)
	{
		CHECK( cudaMemcpy(outputseq[i], array_dev_mutaoutput[i], LEN * sizeof(int), cudaMemcpyDeviceToHost));
	}
	CHECK(cudaMemcpy(outputfit, dev_outputfit, crossCNUM*sizeof(double), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(&crosscount, dev_crosscount, sizeof(int), cudaMemcpyDeviceToHost));
	count=crosscount; 


	for (int i = 0; i < CNUM; i++)
	{
		if (array_dev_input[i])	cudaFree(array_dev_input[i]);
	}
	for (int i = 0; i < crossCNUM; i++)
	{
		if (array_dev_crossoutput[i])	cudaFree(array_dev_crossoutput[i]);
		if (array_dev_mutaoutput[i])	cudaFree(array_dev_mutaoutput[i]);
	}
	for (int i = 0; i < ST; i++)
	{
		if (array_dev_ref[i])	cudaFree(array_dev_ref[i]);
	}
	if (dev_input)	cudaFree(dev_input);
	if (dev_rand)	cudaFree(dev_rand);
	if (dev_inputfit)	cudaFree(dev_inputfit);
	if (dev_outputfit)	cudaFree(dev_outputfit);
	if (dev_ref)	cudaFree(dev_ref);
	if (dev_crossoutput)	cudaFree(dev_crossoutput);
	if (dev_mutaoutput)	cudaFree(dev_mutaoutput);
	if (dev_crosscount)	cudaFree(dev_crosscount);

	return cudaSuccess;
}