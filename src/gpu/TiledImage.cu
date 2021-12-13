
#include "gpu_core_headers.h"

#pragma region Constructors
TiledImage::TiledImage(Image &image_pointer, int wanted_padding, bool verbose)
{
    print_verbose = verbose;
    image = &image_pointer;
    padding = wanted_padding;
    Init();
}


TiledImage::~TiledImage() {
    for(auto &it:tiles) delete it;
    tiles.clear();
}
#pragma endregion Constructors

#pragma region Private Helpers
void TiledImage::Init() {
    dims = make_int3(image->logical_x_dimension, image->logical_y_dimension, image->logical_z_dimension);
    int size_2d = dims.x * dims.y;
    
    // there's no point for using the gpu on small images
    MyAssertTrue(size_2d >= 256, "\tPixel size must be greater than or equal to 256.\n");

    //wxPrintf("\tImage dims: x: %d, y: %d, z: %d\n", dims.x, dims.y, dims.z);

    
    // tile size will be 1024. 32*32
    // or 256. 16*16
    // or 64. 8*8
    // or 16. 4*4
    // this is the tile width (and height) including padding. It should be a value of 2^n for optimal fourier transform calculation
    // it should also be limited to 1024 as this is the maximum block size on the GPU.
    // thus, it will just be 32*32 = 1024
    // unless it smaller than that (then tile size will be 16*16 = 256). Anything smaller than that, is a waste of time to do on the GPU, and is blocked using the above MyAssertTrue
    int tile_width = 32;
    if (size_2d < 1025) {
        tile_width = 16;
    }

    
    // normaly: 1024 = 32 *32
    tile_size = tile_width * tile_width;


    int tile_width_without_padding = tile_width - padding;
    tile_size_without_padding = tile_width_without_padding * tile_width_without_padding;
    
    // wxPrintf("\t padding: %d\n",padding);
    // wxPrintf("\ttile_width_without_padding: %d\n",tile_width_without_padding);
    // determin how many tiles without overlap fit into width. Save the result in number_of_tile_columns
    number_of_tile_columns = ceil(dims.y / tile_width_without_padding);

    // determin how many tiles without overlap fit into height. Save the result in number_of_tile_rows
    number_of_tile_rows = ceil(dims.x / tile_width_without_padding);

    number_of_column_pixels_per_tile =  number_of_row_pixels_per_tile = tile_width_without_padding;

    number_of_tiles =  number_of_tile_rows * number_of_tile_columns * dims.z;
}




// takes image on cpu, and fragment it to multiple CpuImageFragments
void TiledImage::ChunkImage() {
    if (print_verbose) wxPrintf("\tChunkImage function starting.\n");
    if (print_verbose) wxPrintf("\tnumber_of_tile_columns: %d, number_of_tile_rows: %d.\n", number_of_tile_columns, number_of_tile_rows);
    if (print_verbose) wxPrintf("\tnumber_of_column_pixels_per_tile: %d, number_of_row_pixels_per_tile: %d.\n", number_of_column_pixels_per_tile, number_of_row_pixels_per_tile);

    int tile_index = 0;
    int3 wanted_dims = make_int3(number_of_row_pixels_per_tile, number_of_column_pixels_per_tile, 1);
    if (print_verbose) wxPrintf("\tChunkImage function starting loop.\n");
    int pixel_y, pixel_x;

    for (int z = 0; z < dims.z; z++) {
        for (int y = 0; y < number_of_tile_columns; y++) {
            for (int x = 0; x < number_of_tile_rows; x++) {
                pixel_y = number_of_column_pixels_per_tile * y;
                pixel_x = number_of_row_pixels_per_tile * x;

                // if the starting x or y coordinate, plus the size of the tile is bigger than the maximum pixels in the original image: shift the starting pixel to fit.
                // this should only happend in the right-most and bottom tiles
                if (pixel_y + number_of_column_pixels_per_tile >= dims.y) {
                    pixel_y = dims.y - number_of_column_pixels_per_tile;
                }
                if (pixel_x + number_of_column_pixels_per_tile >= dims.x) {
                    pixel_x = dims.x - number_of_row_pixels_per_tile;
                }

                if (print_verbose) wxPrintf("\t\tx: %d y: %d z: %d.\n", pixel_x,pixel_y,z);
         
             
            
                CpuImageFragment* tile = new CpuImageFragment(image, wanted_dims, make_int3(pixel_x,pixel_y,z), padding);

                // Segmentation fault (core dumped) if ran here
                // ProcessTile(tile, print_verbose);
                // PasteTile(tile, print_verbose);

                tiles.push_back(tile);
    
                if (print_verbose) wxPrintf("\t\tTile appended (%d).\n",++tile_index);
             
            }
        }
    }
    if (print_verbose) wxPrintf("\tChunkImage function done.\n");
}

#pragma endregion Private Helpers

#pragma region Public Functions

// a function to test the tiling and assembly process.
// chunks the cpu image to CpuImageFragments, then individually copy to the gpu, ForwardFFT, BackwardFFT, copy to cpu memory and reassembles the tiles into 1 image.
void TiledImage::BlankProcess() {
    if (print_verbose) wxPrintf("\tCutting image to %i tiles.\n", number_of_tiles);
    
    ChunkImage();
    

    //if (print_verbose) wxPrintf("\tProcessing %i tiles.\n", number_of_tiles);

    int tile_index = 0;
    for(auto &tile: tiles) {
        if (print_verbose) wxPrintf("\tTile index: %i.\n", tile_index++);
        
        ProcessTile(tile);
        tile->PasteTile();
    }
    if (print_verbose) wxPrintf("\tTiles processing done.\n");


}
void TiledImage::ProcessTile(CpuImageFragment *tile) {
    if (print_verbose) wxPrintf("\t\tTile processing starting.\n");
    GpuImage gpu_image(tile);
    if (print_verbose) wxPrintf("\t\tGPU image intiated from tile.\n");

    // if (print_verbose) wxPrintf(" tile->is_in_memory: %d.\n", tile->is_in_memory);
    // if (print_verbose) wxPrintf(" gpu_image.is_in_memory: %d.\n", gpu_image.is_in_memory);
   
    // if (print_verbose) wxPrintf(" tile->is_in_memory: %d.\n", tile->is_in_memory);
    // if (print_verbose) wxPrintf(" gpu_image.is_in_memory: %d.\n", gpu_image.is_in_memory);
    // if (print_verbose) wxPrintf(" real_memory_allocated: %d.\n", gpu_image.real_memory_allocated);
    // if (print_verbose) wxPrintf(" is_in_memory: %d.\n", gpu_image.is_in_memory);
    gpu_image.CopyHostToDevice();
    //gpu_image.AddConstant(1.0f);

    if (print_verbose) wxPrintf("\t\tImage copied from host to device.\n");

    gpu_image.ForwardFFT();
    if (print_verbose) wxPrintf("\t\tImage transformed to fourier space.\n");
   
    gpu_image.BackwardFFT();
    if (print_verbose) wxPrintf("\t\tImage transformed to real space.\n");
  
    gpu_image.CopyDeviceToHost(tile);
 

    if (print_verbose) wxPrintf("\t\tImage copied from device to host.\n");
}

#pragma endregion Public Functions