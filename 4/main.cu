/*
	Cource - "Разработка приложений на CUDA "
  	Task 4:
  		При помощи метода наименьших квадратов найти 
  		окружность в изображении. Для каждой случайной 
  		выборки точек организовать их обработку на GPU. 
  		Случайные выборки организовать при помощи 
  		библиотеки CURAND.	

  		Входные данные: изображение размером 640x480 
  		(например, знак ограничения скорости), количество 
  		выборок N, количество элементов в каждой выборке K.

		Выходные данные: изображение с нарисованной на нём 
		окружностью.

		Рекомендация: предварительно к изображению можно 
		применить фильтр Собеля выделения границ и рассматривать 
		точки, у которых нормированное значение цвета >=0.5.
  	Written by Pavel Santaev
*/

#include <stdio.h>
#include <unistd.h> 
#include <math.h>
#include <png.h>
#include <curand.h>
#include <curand_kernel.h>
#include <sys/time.h>

#include <libpng.h>

void abort(const char * s, ...);

__global__ void findCircle(int * pointsX, int * pointsY, 
							int * idx, int length, int K, 
							int * resCircleX, int * resCircleY, int * resRadius, 
							int * resPointCount, long seed){

	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t index = blockDim.x * gridDim.x * y + x;

	curandState_t state;

  	curand_init(seed + index * 9999, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);
  	int * idxLoc = &(idx[index * K]);

  	double xMean = 0;
 	double yMean = 0;

 	double su = 0, sv = 0;
 	double suu = 0, svv = 0, suv = 0;
 	double suuu = 0, svvv = 0, svuu = 0, svvu = 0;

  	for (int i = 0; i < K; i++){
  		float rnd = curand_uniform(&state);	
  		int j = (int)(rnd * length);
  		idxLoc[i] = j;
  		xMean += pointsX[j];
  		yMean += pointsY[j];
  	}

	xMean /= K;
  	yMean /= K;

  	for (int i = 0; i < K; i++){
  		int j = idxLoc[i];

  		double ui = pointsX[j] - xMean;
  		double vi = pointsY[j] - yMean;

  		su += ui;
  		sv += vi;

  		suu += ui * ui;
  		svv += vi * vi;
  		suv += vi * ui;

  		suuu += ui * ui * ui;
  		svvv += vi * vi * vi;
  		svuu += vi * ui * ui;
  		svvu += vi * vi * ui;
  	}

  	double tmpA = 0.5 * (suuu + svvu);
  	double tmpB = 0.5 * (svvv + svuu);
  	double tmpD = tmpB / svv - tmpA * suv / (suu * svv);

  	double vc = tmpD / (1 - suv * suv / (svv * suu));
 	double uc = (tmpA - vc * suv) / suu;
 	double radius = sqrt(uc * uc + vc * vc + (suu + svv) / K);	

 	resRadius[index] = radius;
 	resCircleX[index] = uc + xMean;
 	resCircleY[index] = vc + yMean;

 	int pointsCount = 0;
 	int centerX = uc + xMean;
 	int centerY = vc + yMean;
	for (int i = 0; i < length; i++){
		double treshhold = 2;//radius / 40; // pixels to circle
		double dist = abs( (pointsX[i] - centerX) * (pointsX[i] - centerX) 
			+ (pointsY[i] - centerY) * (pointsY[i] - centerY) - radius * radius);
		if (treshhold * treshhold > dist){
			pointsCount++;
		}
	}
 	//*resCircleX = xMean;
 	//*resCircleY = yMean;
 	resPointCount[index] = pointsCount;
}

__global__ void filter(png_byte * img, png_byte * imgOut, 
					int width, int height, size_t pixelSize){

	double treshhold = 0.2f;

	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (!(x + 1 < width && y + 1 < height && x > 0  && y > 0)){
		return;
	}

	//for (int k = 0; k < 100000; k++){
	const int idx[][2] = 
		{
			{-1, -1}, {0, -1}, {1, -1},
			{-1, 0}, {0, 0}, {1, 0},
			{-1, 1}, {0, 1}, {1, 1}
		};
			
	const float sobel[] = 
		{
			1, 2, 1,
			0, 0, 0,
			-1, -2, -1
		};
	const float sobel2[] = 
		{
			1, 0, -1,
			2, 0, -2,
			1, 0, -1
		};

	png_byte * pxOut = &(imgOut[(y * width + x) * pixelSize]);
	int3 px;
	int3 px2;
	for (int i = 0; i < 9; i++){
		png_byte * pxIn = &(img[((y + idx[i][1]) * width + (x + idx[i][0])) * pixelSize]);
		px.x += (float)sobel[i] * pxIn[0];
		px.y += (float)sobel[i] * pxIn[1];
		px.z += (float)sobel[i] * pxIn[2];

		pxIn = &(img[((y + idx[i][1]) * width + (x + idx[i][0])) * pixelSize]);
		px2.x += (float)sobel2[i] * pxIn[0];
		px2.y += (float)sobel2[i] * pxIn[1];
		px2.z += (float)sobel2[i] * pxIn[2];
	}
	px.x = px.x * px.x;
	px.y = px.y * px.y;
	px.z = px.z * px.z;

	px2.x = px2.x * px2.x;
	px2.y = px2.y * px2.y;
	px2.z = px2.z * px2.z;

	px.x = sqrt((float)(px.x + px2.x));
	px.y = sqrt((float)(px.y + px2.y));
	px.z = sqrt((float)(px.z + px2.z));

	char val = (char)((int)(px.x + px.y + px.z) / 3);

	if ((int)val > treshhold * 255){
		val = (char)255;	
	} else {
		val = (char)0;
	}
	pxOut[0] = val;
	pxOut[1] = val;
	pxOut[2] = val;
	pxOut[3] = 255;
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

int filterPoints(png_bytep * img, int width, int height, 
					size_t pixelSize, int * xs, int *ys){
	int count  = 0;
	for (int i = 1; i < width - 1; i++){
		for (int j = 1; j < height - 1; j++){
			png_byte * pxIn = &(img[j][i * pixelSize]);//&(img[j * width + i * pixelSize]);
			if (pxIn[0] > 200){
				xs[count] = i;
				ys[count] = j;
				count++;
			}
		}
	}
	return count;
}

void drawCircle(png_bytep * img, int width, int height, 
					size_t pixelSize, int x, int y, int rad){

	double PI = 3.1415;
	double angle = 0;
	double delta = PI / (width + height) * 2;
	while (angle < 2 * PI){
		int pxX = (int)(x + sin(angle) * rad);
		int pxY = (int)(y + cos(angle) * rad);
		if (pxY > 0 && pxY < height && pxX > 0 && pxX < width){
			png_byte * pxIn = &(img[pxY][pxX * pixelSize]);	
			pxIn[0] = 255;
			pxIn[1] = 0;
			pxIn[2] = 0;
			pxIn[3] = 255;
		}
		angle += delta;
	}
}

int main(int argc, char ** args){
	//cudaError_t error;
	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep * row_pointers;
	png_bytep * row_pointers_res;

	int K = 10;
	int N = 10;
	// args
	char * file_name;
	if (argc > 3){
		file_name = args[1];
		N = atoi(args[2]);
		K = atoi(args[3]);
	} else {
		abort("You should to add fileName, N, K to args.\n ./out [fileName] [N] [K]");
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
   	
   	//cudaMemcpy(row_pointers, row_pointers_device, size, cudaMemcpyDeviceToHost);

	dim3 threads = dim3(16, 16);
    dim3 blocks  = dim3(ceil(width/(float)threads.x), ceil(height/(float)threads.y));
	filter<<<blocks, threads>>>(row_pointers_device, 
		row_pointers_device_out, width, height, rowSize / width);

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

   	size_t pointSize = width * height * sizeof(int);
	// for points
	int * pointsX = (int *)malloc(pointSize);
	int * pointsY = (int *)malloc(pointSize);
	int pointCount = filterPoints(row_pointers_res, width, height, 
									rowSize / width, pointsX, pointsY);
	printf("Points count = %d\n", pointCount);
	int THREAD_COUNT = N;
	threads = dim3(128, 1);
    blocks  = dim3(ceil(THREAD_COUNT/(float)threads.x), 1);

    int * pointsXdevice;
    int * pointsYdevice;
    int * idxDevice;

	cudaMalloc(&pointsXdevice, pointSize);
	cudaMalloc(&pointsYdevice, pointSize);
	cudaMalloc(&idxDevice, sizeof(int) * K * THREAD_COUNT);

    cudaMemcpy(pointsXdevice, pointsX, pointSize, cudaMemcpyHostToDevice);
    cudaMemcpy(pointsYdevice, pointsY, pointSize, cudaMemcpyHostToDevice);

	int * resCircleXdevice;
    int * resCircleYdevice;
    int * resRadiusdevice;
    int * resPointCountdevice;
	int * resCircleXhost = (int *)malloc(sizeof(int) * THREAD_COUNT);
    int * resCircleYhost = (int *)malloc(sizeof(int) * THREAD_COUNT);
    int * resRadiusdHost = (int *)malloc(sizeof(int) * THREAD_COUNT);
    int * resPointCountHost = (int *)malloc(sizeof(int) * THREAD_COUNT);

	cudaMalloc(&resCircleXdevice, sizeof(int) * THREAD_COUNT);
	cudaMalloc(&resCircleYdevice, sizeof(int) * THREAD_COUNT);
	cudaMalloc(&resRadiusdevice, sizeof(int) * THREAD_COUNT);
	cudaMalloc(&resPointCountdevice, sizeof(int) * THREAD_COUNT);

	// for rand
	struct timeval  tv;
	gettimeofday(&tv, NULL);
	double time_in_mill = 
         (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000;

	findCircle<<<blocks, threads>>>(pointsXdevice, pointsYdevice, 
							idxDevice, pointCount, K, 
							resCircleXdevice, resCircleYdevice, resRadiusdevice, 
							resPointCountdevice, time_in_mill);

	cudaMemcpy(resCircleXhost, resCircleXdevice, sizeof(int) * THREAD_COUNT, cudaMemcpyDeviceToHost);
	cudaMemcpy(resCircleYhost, resCircleYdevice, sizeof(int) * THREAD_COUNT, cudaMemcpyDeviceToHost);
	cudaMemcpy(resRadiusdHost, resRadiusdevice, sizeof(int) * THREAD_COUNT, cudaMemcpyDeviceToHost);
	cudaMemcpy(resPointCountHost, resPointCountdevice, sizeof(int) * THREAD_COUNT, cudaMemcpyDeviceToHost);

	float best = 0;
	int bestIdx = 0;
	for (int i = 0; i < THREAD_COUNT; i++){
		//printf("Res x = %d y = %d r = %d c = %d\n", resCircleXhost[i], 
		//	resCircleYhost[i], resRadiusdHost[i], resPointCountHost[i]);
		if (10 > resRadiusdHost[i]){
			continue;
		}
		float value = resPointCountHost[i] / (resRadiusdHost[i]);
		if (best < value){
			best = value;
			bestIdx = i;
		}
	}
	printf("Best is x = %d y = %d r = %d c = %d idx = %d\n", resCircleXhost[bestIdx], 
		resCircleYhost[bestIdx], resRadiusdHost[bestIdx], resPointCountHost[bestIdx], bestIdx);
	drawCircle(row_pointers_res, width, height, rowSize / width, 
		resCircleXhost[bestIdx], resCircleYhost[bestIdx], resRadiusdHost[bestIdx]);
	drawCircle(row_pointers, width, height, rowSize / width, 
		resCircleXhost[bestIdx], resCircleYhost[bestIdx], resRadiusdHost[bestIdx]);
   	// save png
	savePng("outImg.png", png_ptr, info_ptr, row_pointers_res);
	savePng("outImgOrig.png", png_ptr, info_ptr, row_pointers);


	// free memory
	cudaFree(row_pointers_device);
	cudaFree(row_pointers_device_out);

	cudaFree(resCircleXdevice);
	cudaFree(resCircleYdevice);
	cudaFree(resRadiusdevice);
	cudaFree(resPointCountdevice);

	free(resCircleXhost);
	free(resCircleYhost);
	free(resRadiusdHost);
	free(resPointCountHost);
	/*for (int y=0; y<height; y++){
        free(row_pointers[y]);
    }*/
    //free(row_pointers);
	printf("\nFinished\n");

}


