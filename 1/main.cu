/*
	Cource - "Разработка приложений на CUDA "
  	Task 1:
  		Выделить на GPU массив arr из 10^9 элементов типа 
  		float и инициализировать его с помощью ядра следующим образом: 
  		arr[i] = sin((i%360)*Pi/180). Скопировать массив в память центрального 
  		процессора и посчитать ошибку err = sum_i(abs(sin((i%360)*Pi/180) 
  		- arr[i]))/10^9. Провести исследование зависимости результата от использования 
  		функций: sin, sinf, __sin. Объяснить результат. Проверить результат 
  		при использовании массива типа double.
  	Written by Pavel Santaev
*/

#include <stdio.h>
#include <unistd.h> 
#include <math.h>

typedef double arrType;

__global__ void calcSin(arrType * a, size_t len){
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i = index;
	size_t threadsCount = blockDim.x * gridDim.x;
	while (i < len){
		double value = ((arrType)(i % 360)) * M_PI / 180;
		a[i] = __sinf(value);
		i = i + threadsCount;
	}
}

double calcErr(arrType * arr, size_t len){
	double sum = 0;
	for (int i = 0; i < len; i++){
		sum += abs(sin((i % 360) * M_PI / 180) - arr[i]);
	}
	return sum / len;
}

int main(){
	size_t N = 1000 * 1000 * 100;
	size_t size = sizeof(arrType) * N;
	arrType * ptr;
	cudaError_t error;

	int deviceCount = 0;
	error = cudaGetDeviceCount(&deviceCount);
	if (cudaSuccess != error){
    	printf("Error in cudaGetDeviceCount: %s\n", cudaGetErrorString(error));
    	return 0;
    }
	printf("cudaGetDeviceCount = %x\n", deviceCount);

	int deviceID = 1;
	cudaDeviceProp devProp;
    error = cudaGetDeviceProperties(&devProp, deviceID);
    if (cudaSuccess != error){
    	printf("Error in cudaGetDeviceProperties: %d\n", cudaGetErrorString(error));
    	return 0;
    }
    printf ( "Device %d\n", 0 );
    printf ( "Compute capability     : %d.%d\n", devProp.major, devProp.minor );
    printf ( "Name                   : %s\n", devProp.name );
    printf ( "Total Global Memory    : %d\n", devProp.totalGlobalMem );
    printf ( "Shared memory per block: %d\n", devProp.sharedMemPerBlock );
    printf ( "Registers per block    : %d\n", devProp.regsPerBlock );
    printf ( "Warp size              : %d\n", devProp.warpSize );
    printf ( "Max threads per block  : %d\n", devProp.maxThreadsPerBlock );
    printf ( "Total constant memory  : %d\n", devProp.totalConstMem );
    printf ( "Max Grid Size  : %d %d %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf ( "Max Threads Dim  : %d %d %d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);

	cudaSetDevice(deviceID);

	

	printf("sizeof size_t %d \n", sizeof(size_t));
	printf("sizeof type %d \n", sizeof(arrType));
	printf("allocating %u memory\n", size);
	cudaMalloc(&ptr, size);
	dim3 threads = dim3(devProp.maxThreadsPerBlock, 1);
    dim3 blocks  = dim3(128, 1);
	calcSin<<<blocks, threads>>>(ptr, N);

	int i = 0;
	arrType * hostPtr;
	hostPtr = (arrType *)malloc(size);

	cudaMemcpy(hostPtr, ptr, size, cudaMemcpyDeviceToHost);

	for (i = 0; i < 10; i++){
		printf("%f ", hostPtr[i]);
	}

	printf("\nerror = %0.10f ", calcErr(hostPtr, N));

	cudaFree(ptr);
	free(hostPtr);
	printf("\nfinished\n");
}

