void VectorImage::SetupInitialValues() {

	dims = make_int4(0, 0, 0, 0); pitch = 0;
//   // FIXME: Tempororay for compatibility with the IMage class
//     logical_x_dimension = 0; logical_y_dimension = 0; logical_z_dimension = 0;
	physical_upper_bound_complex = make_int3(0, 0, 0);
	physical_address_of_box_center = make_int3(0, 0, 0);
	physical_index_of_first_negative_frequency = make_int3(0, 0, 0);
	logical_upper_bound_complex = make_int3(0, 0, 0);
	logical_lower_bound_complex = make_int3(0, 0, 0);
	logical_upper_bound_real = make_int3(0, 0, 0);
	logical_lower_bound_real = make_int3(0, 0, 0);

	fourier_voxel_size = make_float3(0.0f, 0.0f, 0.0f);


	number_of_real_space_pixels = 0;


	real_values = NULL;
	complex_values = NULL;

	real_memory_allocated = 0;


	padding_jump_value = 0;

	ft_normalization_factor = 0;

	real_values_gpu = NULL;									// !<  Real array to hold values for REAL images.
	complex_values_gpu = NULL;								// !<  Complex array to hold values for COMP images.


	gpu_plan_id = -1;

	insert_into_which_reconstruction = 0;
	hostImage = NULL;

	cudaErr(cudaEventCreateWithFlags(&nppCalcEvent, cudaEventDisableTiming);)

	cudaErr(cudaGetDevice(&device_idx));
	cudaErr(cudaDeviceGetAttribute(&number_of_streaming_multiprocessors, cudaDevAttrMultiProcessorCount, device_idx));
	limit_SMs_by_threads = 1;

	UpdateBoolsToDefault();
}