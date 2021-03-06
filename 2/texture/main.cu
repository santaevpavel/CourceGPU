/*
	Cource - "Разработка приложений на CUDA "
  	Task 2:
  		Реализовать программу для накладывания фильтров на изображения. 
  		Возможные фильтры: размытие, выделение границ, избавление от шума. 
  		Реализовать два варианта программы, а именно: с применением 
  		разделяемой памяти и текстур. Сравнить время.
		Для работы с графическими файлами рекомендуется использовать libpng (man libpng). 
		Примеры использования библиотеки в /usr/share/doc/libpng12-dev/examples/.
  	Written by Pavel Santaev
*/

#include <stdio.h>
#include <unistd.h> 
#include <math.h>
#include <png.h>

#include <libpng.h>

void abort(const char * s, ...);

texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ png_byte * getPixel(png_byte * img, int w, int h,
							int x, int y, size_t pixelSize){
	int idx = y * w + x;
	return &(img[idx * pixelSize]);
}

__device__ void setPixel(png_byte * pxIn, png_byte * pxOut, 
						size_t pixelSize){
	for (int i = 0; i < pixelSize; i++){
		pxOut[i] = pxIn[i];
	}
}

__device__ void addPixel(png_byte * pxIn, png_byte * pxOut, 
						double alpha, double betta, size_t pixelSize){
	for (int i = 0; i < pixelSize; i++){
		pxOut[i] = (png_byte)(((double)pxOut[i]) * betta +  (((double)pxIn[i]) * alpha));
	}
}

__device__ void setPixelForRobertFilter(
							png_byte * imgOut,
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
			1, 0, 0,
			0, -1, 0,
			0, 0, 0
		};

	png_byte * pxOut = 
		getPixel(imgOut, width, height, x, y, pixelSize);
	png_byte pxOutLoc[32] = {0};
	for (int i = 0; i < 9; i++){
		uchar4 pxIn = tex2D(tex, x + idx[i][0], y + idx[i][1]);
		addPixel((png_byte*)(&pxIn), pxOutLoc, ((double)sobel[i]) / 2, 1, pixelSize);	
	}
	addPixel(pxOutLoc, pxOutLoc, 0, 2, pixelSize);	
	setPixel(pxOutLoc, pxOut, pixelSize);
}

__global__ void filter(png_byte * imgOut, 
					int width, int height, size_t pixelSize){
	size_t blockPxX = width / gridDim.x + 1;
	size_t threadPxY = height / blockDim.x + 1;

	size_t startX = blockPxX * blockIdx.x;
	size_t startY = threadPxY * threadIdx.x;
	for (int i = 0; i < blockPxX; i++){
		for (int j = 0; j < threadPxY; j++){
			int x = startX + i;
			int y = startY + j;
			if (width > x && height > y){
				setPixelForRobertFilter(imgOut, width, height, x, y, pixelSize);

				png_byte * pxOut = getPixel(imgOut, width, 
					height, x, y, pixelSize);

				pxOut[3] = 255;
			}
		}
	}
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

	int deviceID = 0;
	cudaDeviceProp devProp;
    error = cudaGetDeviceProperties(&devProp, deviceID);
    if (cudaSuccess != error){
    	printf("Error in cudaGetDeviceProperties: %d\n", cudaGetErrorString(error));
    	return false;
    }
	cudaSetDevice(deviceID);

	return true;
}

void copyPngToDevice(png_bytep * row_pointers){

}

int main(int argc, char ** args){
	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep * row_pointers;
	png_bytep * row_pointers_res;
	cudaError_t error;

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

	//cudaChannelFormatDesc channelDesc =
	//	cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
 	cudaArray* cuArr;

	if (cudaSuccess != cudaMallocArray(&cuArr, &channelDesc, width, height)){
    	abort("Error: cudaMallocArray");
    }
	
	size_t rowSize = png_get_rowbytes(png_ptr,info_ptr);

	// copy png to device
   	for (int i = 0; i < height; i++){
    	cudaMemcpyToArray(cuArr, 0, i, row_pointers[i], rowSize, cudaMemcpyHostToDevice);
   	}

	// set texture parameters
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false; // access with normalized texture coordinates
	// alloc memory for device
	png_byte * row_pointers_device_out;

	cudaMalloc(&row_pointers_device_out, height * rowSize);

   	
   	// Bind the array to the texture
	error = cudaBindTextureToArray(tex, cuArr, channelDesc);
	if (cudaSuccess != error){
    	abort("Error: cudaBindTextureToArray %s", cudaGetErrorString(error));
    }


	dim3 threads = dim3(16, 1);
    dim3 blocks  = dim3(16, 1);
	filter<<<blocks, threads>>>(row_pointers_device_out, width, height, rowSize / width);


	// copy res png to host
	row_pointers_res = (png_bytep*) malloc(sizeof(png_bytep) * height);

    for (int y = 0; y < height; y++){
		row_pointers_res[y] = 
			(png_byte*) malloc(rowSize);
    }

   	for (int i = 0; i < height; i++){
    	cudaMemcpy(row_pointers_res[i], &(row_pointers_device_out[i * rowSize]),
    		rowSize, cudaMemcpyDeviceToHost);	
   	}

   	// save png
	savePng("outImg.png", png_ptr, info_ptr, row_pointers_res);


	// free memory
	cudaFree(row_pointers_device_out);
	for (int y=0; y<height; y++){
        free(row_pointers[y]);
    }
    free(row_pointers);
    cudaUnbindTexture(tex);
    cudaFreeArray(cuArr);
	printf("\nFinished\n");

}


