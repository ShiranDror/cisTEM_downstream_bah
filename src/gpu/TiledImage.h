
#ifndef TILEDIMAGE_H_
#define TILEDIMAGE_H_

#include <list>
/*
    TiledImage class is used to carve up an image to tiles, aiming to improve performance.

    Currently it only supports GPU
*/


class TiledImage {

    public:
        // Constructor. Sets local pointer to Image and padding. Then calls Init()
        TiledImage(Image &image, int wanted_padding = 3, bool print_verbose = false);

        // Destructor. Deletes all tiles in the tile list
        ~TiledImage();

        // a method to test the tiling process. See method in .cu file for more information.
        void BlankProcess();
    protected:
        // Initialize basic values for the functions of the class: tile width, tile pixel size, how many tiles fit horizontally and vertically in the original image, and how many tiles in total.
        void Init();

        // Populates the list of tiles with CpuImageFragments
        void ChunkImage();

        // The specific processing of each tile, called from BlankProcess(). 
        void ProcessTile(CpuImageFragment *tile);




        int number_of_tiles;
	    std::list<CpuImageFragment*> tiles;


        int3 dims;
        bool is_gpu = true;

        // this a list of optimal pixel sizes for fftw, based on the formula:
        //  2^a * 3^b * 5^c * 7^d * 11^e * 13^f, where e+f is either 0 or 1
        // based on the document on the fftw website: http://www.fftw.org/fftw2_doc/fftw_3.html#:~:text=FFTW%20is%20best%20at%20handling,%2C%20even%20for%20prime%20sizes).
        // values are capped at 1024
        // WARNING: this contains very small numbers that might won't make any sense (e.g. 1)

    private:
        Image *image;
        int tile_size;
        int tile_size_without_padding;
        int padding;
        int number_of_tile_columns;
        int number_of_tile_rows;
        int number_of_column_pixels_per_tile;
        int number_of_row_pixels_per_tile;
        bool print_verbose = false;
};




#endif /* TILEDIMAGE_H_ */