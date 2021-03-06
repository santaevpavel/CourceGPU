/*
	Cource - "Разработка приложений на CUDA "
  	Task 3:
  		Модифицировать предыдущую программу так, чтобы использовались 
  		все имеющиеся в распоряжение программы GPU. Программа должна 
  		определять количество доступных GPU и распределять работу по ним.
  	Written by Pavel Santaev
*/

#include <stdio.h>
#include <unistd.h> 
#include <math.h>
#include <png.h>
#include <multithreading.cpp>

#include <libpng.h>

void abort(const char * s, ...);


__device__ inline png_byte * getPixel(png_byte * img, int w, int h,
							int x, int y, size_t pixelSize){
	int idx = y * w + x;
	return &(img[idx * pixelSize]);
}

__device__ inline void setPixel(png_byte * pxIn, png_byte * pxOut, 
						size_t pixelSize){
	for (int i = 0; i < pixelSize; i++){
		pxOut[i] = pxIn[i];
	}
}

__device__ inline void addPixel(png_byte * pxIn, png_byte * pxOut, 
						float alpha, float betta, size_t pixelSize){
	for (int i = 0; i < pixelSize; i++){
		pxOut[i] = (png_byte)(((double)pxOut[i]) +  (((double)pxIn[i]) * alpha));
	}
}

__device__ inline void setPixelForRobertFilter(
							png_byte * img, png_byte * imgOut,
							int width, int height,
							int x, int y, size_t pixelSize){
	int idx[][2] = 
		{
			{-1, -1}, {0, -1}, {1, -1},
			{-1, 0}, {0, 0}, {1, 0},
			{-1, 1}, {0, 1}, {1, 1}
		};
	int sobel[] = 
		{
			0.025, 0.1, 0.025,
            0.1, 0.5, 0.1,
            0.025, 0.1, 0.025
		};

	png_byte * pxOut = 
		getPixel(imgOut, width, height, x, y, pixelSize);
	png_byte pxOutLoc[4] = {0};
	for (int i = 0; i < 9; i++){
		png_byte * pxIn = 
					getPixel(img, width, height, 
						x + idx[i][0], y + idx[i][1], pixelSize);
		addPixel(pxIn, pxOutLoc, ((double)sobel[i]) / 2, 1, pixelSize);	
	}
	addPixel(pxOutLoc, pxOutLoc, 0, 2, pixelSize);	
	setPixel(pxOutLoc, pxOut, pixelSize);
}

__global__ void filter(png_byte * img, png_byte * imgOut, 
					int width, int height, size_t pixelSize){

	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (!(x <= width && y <= height)){
		return;
	}

	//for (int k = 0; k < 1000; k++){
	const int idx[][2] = 
		{
			{-1, -1}, {0, -1}, {1, -1},
			{-1, 0}, {0, 0}, {1, 0},
			{-1, 1}, {0, 1}, {1, 1}
		};
	const float sobel[] = 
		{
			0.025, 0.1, 0.025,
            0.1, 0.5, 0.1,
            0.025, 0.1, 0.025
		};

	png_byte * pxOut = &(imgOut[(y * width + x) * pixelSize]);
	uint * pxOutInt = (uint *)pxOut;
	png_byte pxOutLoc[4] = {0};
	uint * pxOutLocInt = (uint *) (&pxOutLoc);

	for (int i = 0; i < 9; i++){
		png_byte * pxIn = &(img[((y + idx[i][1]) * width + (x + idx[i][0])) * pixelSize]);
		//uint * pxInInt = (uint *)pxIn;
		addPixel(pxIn, pxOutLoc, ((float)sobel[i]) / 2, 1, 3);	
	}
	*pxOutInt = *pxOutLocInt;
	pxOut[3] = 255;
	//pxOut[3] = k * 0 + 255;
	//}
	//setPixel(pxOutLoc, pxOut, pixelSize);
}

bool initCuda(){
	int deviceCount = 0;
	cudaError_t error;

	error = cudaGetDeviceCount(&deviceCount);
	if (cudaSuccess != error){
    	printf("Error in cudaGetDeviceCount: %s\n", cudaGetErrorString(error));
    	return false;
    }
	printf("cudaGetDeviceCount = %x\n", deviceCount);

	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d.\n",
			device, deviceProp.major, deviceProp.minor);
	}

	int deviceID = 0;
	cudaDeviceProp devProp;
    error = cudaGetDeviceProperties(&devProp, deviceID);
    if (cudaSuccess != error){
    	printf("Error in cudaGetDeviceProperties: %d\n", cudaGetErrorString(error));
    	return false;
    }

	return true;
}

struct ThreadArg{
	size_t device;
	size_t count;
	png_bytep * row_pointers;
	png_bytep * row_pointers_res;
	size_t rowSize;
	size_t height;
	size_t width;
};

static CUT_THREADPROC solverThread(ThreadArg *plan){
	cudaError_t error;
 	// Init GPU
	error = cudaSetDevice(plan->device);
 	if (cudaSuccess != error){
    	printf("Error in cudaSetDevice: %s\n", cudaGetErrorString(error));
    	return;
    }
    size_t height = plan->height;
	size_t width = plan->width;
 	// start kernel
 	dim3 threads = dim3(32, 32);
	dim3 blocks  = dim3(ceil(width/(float)threads.x), ceil(height/(float)threads.y / (plan->count)));

	size_t heightForDevice = blocks.y * threads.y + 2; // for edges
	int yOffet = blocks.y * threads.y * (plan->device) - 1;

	if (yOffet < 0){
		yOffet = 0;
		heightForDevice -= 1;
	}
	png_byte * row_pointers_device;
	png_byte * row_pointers_device_out;
	cudaMalloc(&row_pointers_device, heightForDevice * plan->rowSize);
	cudaMalloc(&row_pointers_device_out, heightForDevice * plan->rowSize);
	for (int i = 0; i < heightForDevice && i < height; i++){
    	cudaMemcpy(&(row_pointers_device[i * plan->rowSize]), (plan->row_pointers)[i + yOffet],
    		plan->rowSize, cudaMemcpyHostToDevice);	
   	}
 	   
    printf("Thread %d: %d\n", plan->device, yOffet);
	filter<<<blocks, threads>>>(row_pointers_device, 
		row_pointers_device_out, width, heightForDevice, plan->rowSize / width);
	printf("Thread %d: filter end\n", plan->device);
	// copy res png to host
   	for (int i = 1 ; (i < heightForDevice - 1) && (yOffet + i < height); i++){
    	cudaMemcpy((plan->row_pointers_res)[yOffet + i], &(row_pointers_device_out[i * plan->rowSize]),
    		plan->rowSize, cudaMemcpyDeviceToHost);	
   	}

 	cudaThreadSynchronize();
 	cudaThreadExit();
 	CUT_THREADEND;
}


int main(int argc, char ** args){
	//cudaError_t error;
	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep * row_pointers;
	png_bytep * row_pointers_res;

	// args
	char * file_name;
	if (argc > 1){
		file_name = args[1];
	} else {
		abort("You should to add fileName to args.\n ./out [fileName]");
	}

	if (!initCuda()){
		return 0;
	}

	openPng(file_name, &png_ptr, &info_ptr, &row_pointers);

	int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);


	// alloc memory for device
	png_byte * row_pointers_device;
	png_byte * row_pointers_device_out;
	size_t rowSize = png_get_rowbytes(png_ptr,info_ptr);
	cudaMalloc(&row_pointers_device, height * rowSize);
	cudaMalloc(&row_pointers_device_out, height * rowSize);
	
	// copy png to device
   	for (int i = 0; i < height; i++){
    	cudaMemcpy(&(row_pointers_device[i * rowSize]), row_pointers[i],
    		rowSize, cudaMemcpyHostToDevice);	
   	}
   	
   	row_pointers_res = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++){
		row_pointers_res[y] = 
			(png_byte*) malloc(rowSize);
    }

	int GPU_N = 0;
	cudaGetDeviceCount(&GPU_N);

	ThreadArg solverOpt[GPU_N];
	CUTThread threadID[GPU_N];
	for(int i = 0; i < GPU_N; i++){
 		solverOpt[i].device = i;
 		solverOpt[i].count = GPU_N;
 		solverOpt[i].row_pointers = row_pointers;
 		solverOpt[i].row_pointers_res = row_pointers_res;
 		solverOpt[i].rowSize = rowSize;
 		solverOpt[i].width = width;
 		solverOpt[i].height = height;
	}
	//Start CPU thread for each GPU
	for(int gpuIndex = 0; gpuIndex < GPU_N; gpuIndex++){
	 	threadID[gpuIndex] = cutStartThread((CUT_THREADROUTINE)solverThread,
 			&solverOpt[gpuIndex]);
	}
	//waiting for GPU results
	cutWaitForThreads(threadID, GPU_N);

   	// save png
	savePng("outImg.png", png_ptr, info_ptr, row_pointers_res);


	// free memory
	cudaFree(row_pointers_device);
	cudaFree(row_pointers_device_out);
	for (int y=0; y<height; y++){
        free(row_pointers[y]);
    }
    free(row_pointers);
	printf("\nFinished\n");

}


