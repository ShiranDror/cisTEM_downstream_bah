class VectorImage {
    public:
    VectorImage();

    protected:
    #pragma region protected

    #pragma region protected variables
      // TODO: These should mostly be made private since they are properties of the data and should not be modified unless a method modifies the data.
	int4 dims;
    // FIXME: Temporary for compatibility with the image class.
    int logical_x_dimension, logical_y_dimension, logical_z_dimension;
	bool 		 is_in_real_space;								// !< Whether the image is in real or Fourier space
	bool 		 object_is_centred_in_box;						//!<  Whether the object or region of interest is near the center of the box (as opposed to near the corners and wrapped around). This refers to real space and is meaningless in Fourier space.
	int3 physical_upper_bound_complex;
	int3 physical_address_of_box_center;
	int3 physical_index_of_first_negative_frequency;
	int3 logical_upper_bound_complex;
	int3 logical_lower_bound_complex;
	int3 logical_upper_bound_real;
	int3 logical_lower_bound_real;

    #pragma endregion protected variables

    #pragma region protected functions
    void SetupInitialValues();


    #pragma endregion protected functions
    #pragma endregion protected
}