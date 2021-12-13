#ifndef CPUIMAGEFRAGMENT_H_
#define CPUIMAGEFRAGMENT_H_
enum PaddingDirection { both, before, after };
enum PaddingMode { circular, repetition, symmetric };
class CpuImageFragment {
    public:
        CpuImageFragment();

        // Constructor. Sets pointer to parent image, set internal values to target copy to and from parent image, and padding size (see FillArrayPadding()).
        CpuImageFragment(Image *image, int3 wanted_dims, int3 starting_coordinates, int padding_size = 3, bool print_verbose = false);
        ~CpuImageFragment();

        // Copies real values from tile to the original image.
        void PasteTile();

        int ReturnReal1DAddressFromPhysicalCoord(int x, int y, int z, int logical_x_dimension, int logical_y_dimension, int padding_jump_value);

        int3 original_coordinates;
        int3 inner_dims;
        int3 outer_dims;
        int padding_jump_value;

        int3 physical_upper_bound_complex;
        int3 physical_address_of_box_center;
        int3 physical_index_of_first_negative_frequency;
        float3 fourier_voxel_size;
        int3 logical_lower_bound_complex;
        int3 logical_upper_bound_complex;
        int3 logical_lower_bound_real;
        int3 logical_upper_bound_real;

        bool is_in_memory = false;
        int real_memory_allocated = 0;
        float *real_values = nullptr;
        bool is_in_real_space = true;
	    bool object_is_centred_in_box = true;

        int matrix_padding_size;
    protected:
   
        void Allocate();
        void Deallocate();
        void UpdateLoopingAndAddressing();

        // copies the relevant (real) values from the parent image to this fragment
        void CopyTileData();
        
        // an adapdation of matlab's pad_array. Used to deal with FFT artefacts in the edge of the tiles
        void FillArrayPadding(float* array, int3 inner_dims, int3 outer_dims, int padding_size = 3, PaddingDirection padding_direction = both, PaddingMode padding_mode = symmetric, float padding_value = 0.0f);


    private:
        Image* image_pointer;
        bool print_verbose = false;
};

#endif /* CPUIMAGEFRAGMENT_H_ */