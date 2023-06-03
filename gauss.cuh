#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<time.h>
#include <windows.h>
using namespace std;

template<class T>
__global__ void division_kernel(T* data, int k, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int element = data[k*N+k];//data[K][K]
    int temp = data[k*N+tid];
    int stride = gridDim.x * blockDim.x;

    while(tid>=k+1&&tid<N){
        data[k*N+tid] = (float)temp/element;
        tid += stride;
    }
    return;
}

template<class T>
__global__ void eliminate_kernel(T*data, int k, int N){
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if(tx==0)
    data[k*N+k]=1.0;

    int row = k+1+blockIdx.x;

    while(row<N){
        int tid = threadIdx.x;
        while(k+1+tid < N){
            int col = k+1+tid;
            T temp_1 = data[(row*N) + col];
            T temp_2 = data[(row*N)+k];
            T temp_3 = data[k*N+col];
            data[(row*N) + col] = temp_1 - temp_2*temp_3;
            tid = tid + blockDim.x;
        }
        __syncthreads();
        if (threadIdx.x == 0){
            data[row * N + k] = 0;
        }
        row += gridDim.x;
    }
    return;
}

template<class T>
class Gauss{
public:
    int width;
    int BLOCK_SIZE;
    T* data;
    T* res;
   
    struct timeval starttime,endtime;
    double timeuse;

    Gauss(int N, int block_size){
        width = N;
        BLOCK_SIZE = block_size;
        data = new T[N*N];
        res = new T[N*N];
        
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                data[i*N+j] = rand()%100;
                res[i*N+j] = data[i*N+j];
            }
        }

        srand(time(NULL));
    }

    void simple_lu();
    void cuda_lu(int grid_w);

    ~Gauss(){
        delete[] data;
        delete[] res;
    }

};

template<class T>
void Gauss<T>::simple_lu(){
    long long head, tail, freq;
    QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&freq));
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&head));
    for (int k = 0; k < width; k++)
    {  
		for (int j = k + 1; j < width; j++)
            data[k*width+j] = data[k*width+j] / data[k*width+k];

        data[k*width+k] = 1.0;
		
		for (int i = k + 1; i < width; i++)
        {
            for (int j = k + 1; j < width; j++)
                data[i*width+j] = data[i*width+j] - data[i*width+k] * data[k*width+j];
			
			data[i*width+k] = 0;
        }
		
    }

    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&tail));
    cout << "Col: "<< (tail - head) * 1000.0 / freq << "ms" << endl;
}

template<class T>
void Gauss<T>::cuda_lu(int grid_w){
    dim3 grid = dim3(grid_w,1,1);
    dim3 block = dim3(BLOCK_SIZE,1,1);
    
    T* data_D;
    //T* res_D;

    cudaError_t ret;
    cudaEvent_t start, stop;
	float elapsedTime = 0.0;

    ret = cudaMalloc(&data_D, sizeof(T)*width*width);
    if(ret!=cudaSuccess){
		printf("cudaMalloc gpudata failed!\n");
	}

    ret = cudaMemcpy(data_D,res,sizeof(T)*width*width,cudaMemcpyHostToDevice);
    if(ret!=cudaSuccess){
        printf("cudaMemcpyHostToDevice failed!\n");
    }

    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    for(int k=0;k<width;k++){

        division_kernel<<<grid,block>>>(data_D, k, width);
        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        if(ret!=cudaSuccess){
            printf("division_kernel failed, %s\n",cudaGetErrorString(ret));
        }
    
        //cudaMemset((data_D+k*width+k), 1 ,sizeof(T));

        eliminate_kernel<<<grid,block>>>(data_D, k, width);
        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        if(ret!=cudaSuccess){
            printf("eliminate_kernel failed, %s\n",cudaGetErrorString(ret));
        }

    }

    ret = cudaMemcpy(res,data_D,sizeof(T)*width*width,cudaMemcpyDeviceToHost);
    if(ret!=cudaSuccess){
        printf("cudaMemcpyDeviceToHost failed!\n");
    }

    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("GPU_LU: %f ms\n", elapsedTime);

    cudaEventDestroy(start);
	cudaEventDestroy(stop);
    cudaFree(data_D);
    return;
}


