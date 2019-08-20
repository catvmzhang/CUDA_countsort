#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define LISTSIZE 10000000
#define MAXNUM 10000
#define THREAD_PER_BLOCK 1024

__global__ void gpu_countsort(int* globalTable_d, int* unsort){
	__shared__ int table[MAXNUM];
	if(threadIdx.x == 0) memset(table, 0, sizeof(int)*MAXNUM);
	__syncthreads();//block level synchronization
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < LISTSIZE){
		int num = unsort[index];
		atomicAdd(&table[num-1], 1);
	}
	__syncthreads();
	if(threadIdx.x == 0){
		for(int i=0; i<MAXNUM; i++){
			atomicAdd(&(globalTable_d[i]), table[i]);
		}
	}
}

void genList(int** unsort){
	*unsort = (int*)malloc(sizeof(int) * LISTSIZE);
	for(int i=0; i<LISTSIZE; i++){
		(*unsort)[i] = rand()%MAXNUM + 1;
	}
}

int main()
{
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int *unsort;
	genList(&unsort);

	int *unsort_d, *table_d;
	int listSize = LISTSIZE * sizeof(int);
	int tableSize = MAXNUM * sizeof(int);
	cudaMalloc((void**)&unsort_d, listSize);
	cudaMemcpy(unsort_d, unsort, listSize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&table_d, tableSize);
	cudaMemset(table_d, 0, tableSize);

	int blockNum;
	blockNum = (LISTSIZE/THREAD_PER_BLOCK) + ( LISTSIZE%THREAD_PER_BLOCK==0 ?0:1 );
	cudaEventRecord(start, 0);
	gpu_countsort<<<blockNum, THREAD_PER_BLOCK>>>(table_d, unsort_d);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaDeviceSynchronize();
	printf("time in gpu: %31.f ms\n", time);

	int *table, *sort;
	sort = (int*)malloc(listSize);
	memset(sort, 0, listSize);
	table = (int*)malloc(tableSize);
	cudaMemcpy(table, table_d, tableSize, cudaMemcpyDeviceToHost);

	int index=0;
	for(int i=0; i<MAXNUM; i++){
		for(int j=0; j<table[i]; j++) sort[index++] = i+1;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("time in cpu: %31.f ms\n", time);
//	for(int i=0; i<LISTSIZE; i++) printf("%d ", sort[i]);

	cudaFree(unsort_d);
	cudaFree(table_d);
	free(unsort);
	free(table);

	return 0;
}
