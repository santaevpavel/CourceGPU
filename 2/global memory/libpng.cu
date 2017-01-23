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
#include <libpng.h>

void abort(const char * s, ...){
    va_list args;
    va_start(args, s);
    vfprintf(stdout, s, args);
    fprintf(stdout, "\n");
    va_end(args);
    abort();
}

void savePng(const char* file_name, 
			png_structp png_ptr_in,
			png_infop info_ptr_in,
			png_bytep *row_pointers){
    /* create file */
    png_structp png_ptr;
    png_infop info_ptr;

	int width = png_get_image_width(png_ptr_in, info_ptr_in);
    int height = png_get_image_height(png_ptr_in, info_ptr_in);
    png_byte color_type = png_get_color_type(png_ptr_in, info_ptr_in);
    png_byte bit_depth = png_get_bit_depth(png_ptr_in, info_ptr_in);

    FILE *fp = fopen(file_name, "wb");
    if (!fp){
        abort("[write_png_file] File %s could not be opened for writing", file_name);
    }

    /* initialize stuff */
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr){
        abort("[write_png_file] png_create_write_struct failed");
    }
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr){
        abort("[write_png_file] png_create_info_struct failed");
    }

    if (setjmp(png_jmpbuf(png_ptr))){
        abort("[write_png_file] Error during init_io");
    }

    png_init_io(png_ptr, fp);


    /* write header */
    if (setjmp(png_jmpbuf(png_ptr))){
        abort("[write_png_file] Error during writing header");
    }

    png_set_IHDR(png_ptr, info_ptr, width, height,
                bit_depth, color_type, PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr))){
        abort("[write_png_file] Error during writing bytes");
    }

    png_write_image(png_ptr, row_pointers);


    /* end write */
    if (setjmp(png_jmpbuf(png_ptr))){
        abort("[write_png_file] Error during end of write");
    }

    png_write_end(png_ptr, NULL);

    /* cleanup heap allocation */
    for (int y=0; y<height; y++){
        free(row_pointers[y]);
    }
    free(row_pointers);

    printf("File %s is saved\n", file_name);
    fclose(fp);
}

void openPng(const char* file_name, 
	png_structp * out_png_ptr,
	png_infop * out_info_ptr,
	png_bytep ** out_row_pointers){

	png_structp png_ptr;
	png_infop info_ptr;
	png_bytep * row_pointers;

	png_bytep header = (png_bytep)malloc(8 * 8);    // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(file_name, "rb");
    if (!fp){
        abort("[read_png_file] File %s could not be opened for reading", file_name);
    }
    fread(header, 1, 8, fp);    
    if (png_sig_cmp(header, 0, 8)){
    	fclose(fp);
        abort("[read_png_file] File %s is not recognized as a PNG file", file_name);
    }
    
	/* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr){
		fclose(fp);
		abort("[read_png_file] png_create_read_struct failed");
	}

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr){
    	fclose(fp);
        abort("[read_png_file] png_create_info_struct failed");
	}

	if (setjmp(png_jmpbuf(png_ptr))){
		fclose(fp);
		abort("[read_png_file] Error during init_io");
	}

	printf("PNG file is opened\n");

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	printf("\t  width = %d\n", width);
	printf("\t  height = %d\n", height);

    int number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

	

    /* read file */
    if (setjmp(png_jmpbuf(png_ptr))){
        abort("[read_png_file] Error during read_image");
    }

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);

    for (int y = 0; y < height; y++){
		row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));
    }
    png_read_image(png_ptr, row_pointers);

    fclose(fp);

	*out_png_ptr = png_ptr;
	*out_info_ptr = info_ptr;
	*out_row_pointers = row_pointers;
}

