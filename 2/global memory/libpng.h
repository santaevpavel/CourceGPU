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


void openPng(const char* file_name, 
            png_structp * out_png_ptr,
            png_infop * out_info_ptr,
            png_bytep ** out_row_pointers);

void savePng(const char* file_name, 
            png_structp png_ptr_in,
            png_infop info_ptr_in,
            png_bytep *row_pointers);
