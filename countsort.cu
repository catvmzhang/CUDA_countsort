#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define LISTSIZE 1000000
#define MAXNUM 10000
#define THREAD_PER_BLOCK 256

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
	gpu_countsort<<<blockNum, THREAD_PER_BLOCK>>>(table_d, unsort_d);

	cudaDeviceSynchronize();

	int *table, *sort;
	sort = (int*)malloc(listSize);
	memset(sort, 0, listSize);
	table = (int*)malloc(tableSize);
	cudaMemcpy(table, table_d, tableSize, cudaMemcpyDeviceToHost);

	int index=0;
	for(int i=0; i<MAXNUM; i++){
		for(int j=0; j<table[i]; j++) sort[index++] = i+1;
	}

//	for(int i=0; i<LISTSIZE; i++) printf("%d ", sort[i]);

	cudaFree(unsort_d);
	cudaFree(table_d);
	free(unsort);
	free(table);

	return 0;
}
