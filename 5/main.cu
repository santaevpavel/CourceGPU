/*
	Cource - "Разработка приложений на CUDA "
  	Task 5:

  	Written by Pavel Santaev
*/

#include <stdio.h>
#include <unistd.h> 
#include <math.h>
#include <png.h>

#include <libpng.h>
#include "Shape.h"
#include "Scene.h"
#include "Sphere.h"
#include "Color.h"

void abort(const char * s, ...);

__global__ void filter(png_byte * img, png_byte * imgOut, 
					int width, int height, size_t pixelSize){//, Scene * scene, Shape ** shapes){

	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (!(x < width && y < height && x >= 0  && y >= 0)){
		return;
	}
	//scene->shapes = shapes;
	Shape * sphere = new Sphere(Point3D(10, 0, 0), 2);
	Shape * sphere2 = new Sphere(Point3D(10, 3, 1), 1);
	Shape ** shapes = new Shape*[2];
	shapes[0] = sphere;
	shapes[1] = sphere2;
	Scene * scene = new Scene(shapes, 2, Point3D(10, 7, 0));

	for (int x = 0; x < width; x++){
		for (int y = 0; y < height; y++){
			Vector3D camDir(1, 0, 0);
			Vector3D camDirX(0, 1.0*width/height, 0);
			Vector3D camDirY(0, 0, 1);
			Vector3D pxDir = camDir.add(camDirX*(-0.5 + 1.0 * x / width)).add(camDirY*(-0.5 + 1.0 * y / height));
			Line line(Point3D(0, 0, 0), pxDir);
			Color color = scene->trace(line);

			png_byte * pxOut = &(imgOut[(y * width + x) * pixelSize]);

			pxOut[0] = color.r;
			pxOut[1] = color.g;
			pxOut[2] = color.b;
			pxOut[3] = 255;
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

	int deviceID = 1;
	cudaDeviceProp devProp;
    error = cudaGetDeviceProperties(&devProp, deviceID);
    if (cudaSuccess != error){
    	printf("Error in cudaGetDeviceProperties: %d\n", cudaGetErrorString(error));
    	return false;
    }
	cudaSetDevice(deviceID);

	return true;
}

int main(int argc, char ** args){
	//cudaError_t error;
	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep * row_pointers;
	png_bytep * row_pointers_res;

	// args
	char * file_name;
	file_name = args[1];
	/*if (argc > 3){
		file_name = args[1];
		N = atoi(args[2]);
		K = atoi(args[3]);
	} else {
		abort("You should to add fileName, N, K to args.\n ./out [fileName] [N] [K]");
	}*/

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

	/*Shape * sphere = new Sphere(Point3D(10, 0, 0), 2);
	Shape ** shapes = new Shape*[1];
	shapes[0] = sphere;

	Shape * sphereDevice;
	cudaMalloc(&sphereDevice, sizeof(Sphere));
	cudaMemcpy(sphereDevice, sphere, sizeof(Sphere), cudaMemcpyHostToDevice);

	Shape ** shapesDevice;
	cudaMalloc(&shapesDevice, sizeof(Shape*));
	cudaMemcpy(&(shapesDevice[0]), &sphereDevice, sizeof(Shape*), cudaMemcpyHostToDevice);

	Scene scene(nullptr, 1, Point3D(10, 5, 0));

	Scene * sceneDevice;
	cudaMalloc(&sceneDevice, sizeof(Scene));
	cudaMemcpy(sceneDevice, &scene, sizeof(Scene), cudaMemcpyHostToDevice);*/

	dim3 threads = dim3(1, 1);
    dim3 blocks  = dim3(1, 1);
    //dim3 threads = dim3(16, 16);
	//dim3 blocks  = dim3(ceil(width/(float)threads.x), ceil(height/(float)threads.y));
	filter<<<blocks, threads>>>(row_pointers_device, 
		row_pointers_device_out, width, height, rowSize / width);//, sceneDevice, shapesDevice);

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

	savePng("outImg.png", png_ptr, info_ptr, row_pointers_res);

	// free memory
	cudaFree(row_pointers_device);
	cudaFree(row_pointers_device_out);


	//free(resCircleXhost);
	for (int y=0; y<height; y++){
        free(row_pointers[y]);
    }
    free(row_pointers);
	printf("\nFinished\n");
	return 0;
}


