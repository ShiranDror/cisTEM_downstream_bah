#include "gpu_core_headers.h"

#pragma region Constructors
CpuImageFragment::CpuImageFragment() {}
CpuImageFragment::CpuImageFragment(Image *image, int3 wanted_dims, int3 starting_coordinates, int padding_size, bool print_verbose) {
    print_verbose = print_verbose;
    image_pointer = image;
    inner_dims = wanted_dims;
    matrix_padding_size = padding_size;
    Allocate();
    original_coordinates = starting_coordinates;
    

    CopyTileData();
    FillArrayPadding(real_values, inner_dims, outer_dims, padding_size, both, symmetric, 1.0f);
}

CpuImageFragment::~CpuImageFragment() {
    Deallocate();
}
#pragma endregion Constructors

#pragma region Private Methods
void CpuImageFragment::Allocate()
{
    MyDebugAssertTrue(inner_dims.x > 0 && inner_dims.y > 0 && inner_dims.z > 0,"Bad dimensions: %i %i %i\n",inner_dims.x,inner_dims.y,inner_dims.z);

	// check to see if we need to do anything?

	if (is_in_memory == true)
	{
		Deallocate();
	}

	// if we got here we need to do allocation..
    
	outer_dims = make_int3(inner_dims.x + matrix_padding_size, inner_dims.y + matrix_padding_size, inner_dims.z);

	// first_x_dimension
	if (IsEven(outer_dims.x) == true) real_memory_allocated =  outer_dims.x / 2 + 1;
	else real_memory_allocated = (outer_dims.x - 1) / 2 + 1;

	real_memory_allocated *= outer_dims.y * outer_dims.z; // other dimensions
	real_memory_allocated *= 2; // room for complex

	real_values = new float[real_memory_allocated]; //(float *) fftwf_malloc(sizeof(float) * real_memory_allocated);
	//complex_values = (std::complex<float>*) real_values;  // Set the complex_values to point at the newly allocated real values;

	is_in_memory = true;

	// Update addresses etc..

    UpdateLoopingAndAddressing();

    // set the loop junk value..

	padding_jump_value = IsEven(outer_dims.x) ? 2 : 1;
}
void CpuImageFragment::Deallocate()
{
	if (is_in_memory)
	{
		fftwf_free(real_values);
		is_in_memory = false;
	}
}
void CpuImageFragment::UpdateLoopingAndAddressing()
{
	physical_upper_bound_complex = make_int3(outer_dims.x / 2, outer_dims.y - 1, outer_dims.z - 1);
    physical_address_of_box_center = make_int3(outer_dims.x / 2, outer_dims.y / 2, outer_dims.z / 2);

    physical_index_of_first_negative_frequency = make_int3(0, outer_dims.y / 2, outer_dims.z / 2);
    if (IsEven(outer_dims.y) == false) physical_index_of_first_negative_frequency.y = outer_dims.y / 2 + 1;
    if (IsEven(outer_dims.z) == false) physical_index_of_first_negative_frequency.z = outer_dims.z / 2 + 1;

    // Update the Fourier voxel size
    fourier_voxel_size = make_float3(1.0 / double(outer_dims.x), 1.0 / double(outer_dims.y), 1.0 / double(outer_dims.z));

    logical_lower_bound_complex = make_int3(
                                            // x:
                                            IsEven(outer_dims.x) ? -outer_dims.x / 2 : -(outer_dims.x-1) / 2,
                                            //y:
                                            IsEven(outer_dims.y) ? -outer_dims.y / 2 : -(outer_dims.y-1) / 2,
                                            //z:
                                            IsEven(outer_dims.z) ? -outer_dims.z / 2 :  -(outer_dims.z - 1) / 2
                                            );

    logical_upper_bound_complex = make_int3(
                                            // x:
                                            IsEven(outer_dims.x) ? outer_dims.x / 2 : (outer_dims.x-1) / 2,
                                            //y:
                                            IsEven(outer_dims.y) ? outer_dims.y / 2 - 1 : (outer_dims.y-1) / 2,
                                            //z:
                                            IsEven(outer_dims.z) ? outer_dims.z / 2 - 1 :  (outer_dims.z - 1) / 2
                                            );
    logical_lower_bound_real = make_int3(
                                            // x:
                                            IsEven(outer_dims.x) ? -outer_dims.x / 2 : -(outer_dims.x-1) / 2,
                                            //y:
                                            IsEven(outer_dims.y) ? -outer_dims.y / 2 : -(outer_dims.y-1) / 2,
                                            //z:
                                            IsEven(outer_dims.z) ? -outer_dims.z / 2 :  -(outer_dims.z - 1) / 2
                                            );

    logical_upper_bound_real = make_int3(
                                            // x:
                                            IsEven(outer_dims.x) ? outer_dims.x / 2 - 1 : (outer_dims.x-1) / 2,
                                            //y:
                                            IsEven(outer_dims.y) ? outer_dims.y / 2 - 1 : (outer_dims.y-1) / 2,
                                            //z:
                                            IsEven(outer_dims.z) ? outer_dims.z / 2 - 1 :  (outer_dims.z - 1) / 2
                                            );

}

void CpuImageFragment::FillArrayPadding(float* array, int3 inner_dims, int3 outer_dims, int padding_size, PaddingDirection padding_direction, PaddingMode padding_mode, float padding_value) { // PaddingDirection padding_direction = both, PaddingMode padding_mode = symmetric)
//     let shift_x = image.padding;
//     let shift_y = image.padding;

//     const right_border = image.dims.y + image.padding - 1;
//     const bottom_border = image.dims.x + image.padding - 1;

//     // shift_x and y are variables that point to the first row and column of the actual data in the array that needs padding (and therefore should not be overwritten)
//     if (padding_direction == "after") {
//         shift_x = 0;
//         shift_y = 0;
//     }

    int shift_x = padding_size;
    int shift_y = padding_size;

    int right_border = inner_dims.y + padding_size - 1;
    int bottom_border = inner_dims.x + padding_size - 1;

    // shift_x and y are variables that point to the first row and column of the actual data in the array that needs padding (and therefore should not be overwritten)
    if (padding_direction == after) {
        shift_x = 0;
        shift_y = 0;
    }

    
    int source_x, source_y, destination_index, source_index;

    for (int x = 0; x < outer_dims.x; x++) {
        for (int y = 0; y < outer_dims.y; y++) {
            if (x < shift_x || y < shift_y || y > right_border || x > bottom_border) {
                destination_index = ReturnReal1DAddressFromPhysicalCoord(x, y, 0, outer_dims.x, outer_dims.y, padding_jump_value);
                switch (padding_mode) {

                    case repetition:
                     
                        array[destination_index] = padding_value;
                        break;
                    case circular:
                        source_x = x;
                        source_y = y;
                        if (x < shift_x) source_x = bottom_border + 1 - (shift_x - x);
                        else if (x > bottom_border) source_x = shift_x + (x - bottom_border - 1);
                        

                        if (y < shift_y) source_y = right_border + 1 - (shift_y - y);
                        else if (y > right_border) source_y = shift_y + (y - right_border - 1);

                        source_index = ReturnReal1DAddressFromPhysicalCoord(source_x, source_y, 0, outer_dims.x, outer_dims.y, padding_jump_value);
                        array[destination_index] = array[source_index];
                    case symmetric:
                        source_x = x;
                        source_y = y;

                        if (x < shift_x) source_x = shift_x + (shift_x - x);
                        else if (x > bottom_border) source_x = bottom_border - (x - bottom_border);
                        

                        if (y < shift_y) source_y = shift_y + (shift_y - y);
                        else if (y > right_border) source_y = right_border - (y - right_border);

                        source_index = ReturnReal1DAddressFromPhysicalCoord(source_x, source_y, 0, outer_dims.x, outer_dims.y, padding_jump_value);
                        array[destination_index] = array[source_index]; 

                }
                
            }
            // for testing purposes only
            // else {
            //     destination_index = ReturnReal1DAddressFromPhysicalCoord(x, y, 0, outer_dims.x, outer_dims.y, padding_jump_value);
            //     array[destination_index] = array[destination_index] + 1.0f;
            // }
        }
    }
}


void CpuImageFragment::CopyTileData() {
 
    int3 wanted_dims = inner_dims;
    int3 starting_coordinates  = original_coordinates;
    Image* image = image_pointer;
   
    if (print_verbose) wxPrintf("\tCopyTileData function starting.\n");

    if (print_verbose) wxPrintf("\tCopyTileData function starting loop.\n");
    // if (print_verbose) wxPrintf("\t\tstarting_coordinates x: %d, y: %d, z: %d.\n",starting_coordinates.x,starting_coordinates.y,starting_coordinates.z,i);
    // if (print_verbose) wxPrintf("\t\twanted_dims x: %d, y: %d, z: %d.\n",wanted_dims.x,wanted_dims.y,wanted_dims.z,i);
    int destination_index, source_index;
    int z = starting_coordinates.z;

        for (int y = 0; y < wanted_dims.y; y++)
        {
            for (int x = 0; x < wanted_dims.x; x++) 
            {
                
                source_index =      ReturnReal1DAddressFromPhysicalCoord(x+starting_coordinates.x,         y+starting_coordinates.y,            z, image->logical_x_dimension,  image->logical_y_dimension, image->padding_jump_value   );
                destination_index = ReturnReal1DAddressFromPhysicalCoord(x+matrix_padding_size,            y+matrix_padding_size,               0, outer_dims.x,                outer_dims.y,               padding_jump_value          );
                //if (print_verbose) wxPrintf("\t\tx: %d, y: %d, z: %d, i: %d.\n",x,y,z,i);
                real_values[destination_index] = image->real_values[source_index]; // perhaps adding a constant here to test?
            }
        }
    

    if (print_verbose) wxPrintf("\tCopyTileData function done.\n");
}

void CpuImageFragment::PasteTile() {
    Image* image = image_pointer;
    int source_shift_x = matrix_padding_size;
    int source_shift_y = matrix_padding_size;


    int source_index, destination_index;

  
    int destination_x, destination_y, source_x, source_y;
    int source_z = 0;
    int destination_z = original_coordinates.z;

    for (int y = 0; y < inner_dims.y; y++)
    {
        for (int x = 0; x < inner_dims.x; x++)
        {
            source_x = x + source_shift_x;
            source_y = y + source_shift_y;
            source_index =      ReturnReal1DAddressFromPhysicalCoord(source_x,      source_y,       source_z,       outer_dims.x,                 outer_dims.y,                  padding_jump_value       );


            destination_x = x + original_coordinates.x;
            destination_y = y + original_coordinates.y;
            destination_index = ReturnReal1DAddressFromPhysicalCoord(destination_x, destination_y,  destination_z,  image->logical_x_dimension,   image->logical_y_dimension,    image->padding_jump_value);
            
            image->real_values[destination_index] = real_values[source_index];
        }
    }
}




int CpuImageFragment::ReturnReal1DAddressFromPhysicalCoord(int x, int y, int z, int logical_x_dimension, int logical_y_dimension, int padding_jump_value)
{
    return (((logical_x_dimension + padding_jump_value) * logical_y_dimension) * z) + ((logical_x_dimension + padding_jump_value) * y) + x;
};
#pragma endregion Private Methods