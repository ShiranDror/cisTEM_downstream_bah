#include "gpu_core_headers.h"

#pragma region public

#pragma region constructors
VectorImage::VectorImage()
{ 
  SetupInitialValues();
}
VectorImage::VectorImage(const VectorImage &other_image) // copy constructor
{
	SetupInitialValues();
	*this = other_image;
}
VectorImage::~VectorImage() 
{
	Deallocate();
}
#pragma endregion constructors

// deep copy
VectorImage Clone() {
    VectorImage* clone = new VectorImage(*this);
    return clone;
}

#pragma endregion public


#pragma region protected
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

void VectorImage::UpdateBoolsToDefault()
{
	is_meta_data_initialized = false;

	is_in_memory = false;
	is_in_real_space = true;
	object_is_centred_in_box = true;
	image_memory_should_not_be_deallocated = false;

	is_in_memory_gpu = false;
	is_host_memory_pinned = false;

	// libraries
	is_fft_planned = false;
    //	is_cublas_loaded = false;
	is_npp_loaded = false;

	// Buffers
	is_allocated_image_buffer = false;
	is_allocated_mask_CSOS = false;

	is_allocated_sum_buffer = false;
	is_allocated_min_buffer = false;
	is_allocated_minIDX_buffer = false;
	is_allocated_max_buffer = false;
	is_allocated_maxIDX_buffer = false;
	is_allocated_minmax_buffer = false;
	is_allocated_minmaxIDX_buffer = false;
	is_allocated_mean_buffer = false;
	is_allocated_meanstddev_buffer = false;
	is_allocated_countinrange_buffer = false;
	is_allocated_l2norm_buffer = false;
	is_allocated_dotproduct_buffer = false;
	is_allocated_16f_buffer = false;

	// Callbacks
	is_set_convertInputf16Tof32 = false;
	is_set_scaleFFTAndStore = false;
	is_set_complexConjMulLoad = false;
	is_allocated_clip_into_mask = false;
	is_set_realLoadAndClipInto = false;

}

void VectorImage::UpdateLoopingAndAddressing(int wanted_x_size, int wanted_y_size, int wanted_z_size)
{


	dims.x = wanted_x_size;
	dims.y = wanted_y_size;
	dims.z = wanted_z_size;

	physical_address_of_box_center.x = wanted_x_size / 2;
	physical_address_of_box_center.y= wanted_y_size / 2;
	physical_address_of_box_center.z= wanted_z_size / 2;

	physical_upper_bound_complex.x= wanted_x_size / 2;
	physical_upper_bound_complex.y= wanted_y_size - 1;
	physical_upper_bound_complex.z= wanted_z_size - 1;


	//physical_index_of_first_negative_frequency.x= wanted_x_size / 2 + 1;
	if (IsEven(wanted_y_size) == true)
	{
		physical_index_of_first_negative_frequency.y= wanted_y_size / 2;
	}
	else
	{
		physical_index_of_first_negative_frequency.y= wanted_y_size / 2 + 1;
	}

	if (IsEven(wanted_z_size) == true)
	{
		physical_index_of_first_negative_frequency.z= wanted_z_size / 2;
	}
	else
	{
		physical_index_of_first_negative_frequency.z= wanted_z_size / 2 + 1;
	}


    // Update the Fourier voxel size

	fourier_voxel_size.x= 1.0 / double(wanted_x_size);
	fourier_voxel_size.y= 1.0 / double(wanted_y_size);
	fourier_voxel_size.z= 1.0 / double(wanted_z_size);

	// Logical bounds
	if (IsEven(wanted_x_size) == true)
	{
		logical_lower_bound_complex.x= -wanted_x_size / 2;
		logical_upper_bound_complex.x=  wanted_x_size / 2;
	    logical_lower_bound_real.x   = -wanted_x_size / 2;
	    logical_upper_bound_real.x   =  wanted_x_size / 2 - 1;
	}
	else
	{
		logical_lower_bound_complex.x= -(wanted_x_size-1) / 2;
		logical_upper_bound_complex.x=  (wanted_x_size-1) / 2;
		logical_lower_bound_real.x   = -(wanted_x_size-1) / 2;
		logical_upper_bound_real.x    =  (wanted_x_size-1) / 2;
	}


	if (IsEven(wanted_y_size) == true)
	{
	    logical_lower_bound_complex.y= -wanted_y_size / 2;
	    logical_upper_bound_complex.y=  wanted_y_size / 2 - 1;
	    logical_lower_bound_real.y   = -wanted_y_size / 2;
	    logical_upper_bound_real.y   =  wanted_y_size / 2 - 1;
	}
	else
	{
	    logical_lower_bound_complex.y= -(wanted_y_size-1) / 2;
	    logical_upper_bound_complex.y=  (wanted_y_size-1) / 2;
	    logical_lower_bound_real.y   = -(wanted_y_size-1) / 2;
	    logical_upper_bound_real.y    =  (wanted_y_size-1) / 2;
	}

	if (IsEven(wanted_z_size) == true)
	{
		logical_lower_bound_complex.z= -wanted_z_size / 2;
		logical_upper_bound_complex.z=  wanted_z_size / 2 - 1;
		logical_lower_bound_real.z   = -wanted_z_size / 2;
		logical_upper_bound_real.z   =  wanted_z_size / 2 - 1;

	}
	else
	{
		logical_lower_bound_complex.z= -(wanted_z_size - 1) / 2;
		logical_upper_bound_complex.z=  (wanted_z_size - 1) / 2;
		logical_lower_bound_real.z   = -(wanted_z_size - 1) / 2;
		logical_upper_bound_real.z    =  (wanted_z_size - 1) / 2;
	}
}

void VectorImage::Allocate(int wanted_x_size, int wanted_y_size, int wanted_z_size, bool should_be_in_real_space)
{

	MyAssertTrue(wanted_x_size > 0 && wanted_y_size > 0 && wanted_z_size > 0,"Bad dimensions: %i %i %i\n",wanted_x_size,wanted_y_size,wanted_z_size);

	// check to see if we need to do anything?

	if (is_in_memory_gpu == true)
	{
		is_in_real_space = should_be_in_real_space;
		if (wanted_x_size == dims.x && wanted_y_size == dims.y && wanted_z_size == dims.z)
		{
			// everything is already done..
			is_in_real_space = should_be_in_real_space;
	//			wxPrintf("returning\n");
			return;
		}
		else
		{
		  Deallocate();
		}
	}

	SetupInitialValues();
	this->is_in_real_space = should_be_in_real_space;
	dims.x = wanted_x_size; dims.y = wanted_y_size; dims.z = wanted_z_size;

	// if we got here we need to do allocation..

	// first_x_dimension
	if (IsEven(wanted_x_size) == true) real_memory_allocated =  wanted_x_size / 2 + 1;
	else real_memory_allocated = (wanted_x_size - 1) / 2 + 1;

	real_memory_allocated *= wanted_y_size * wanted_z_size; // other dimensions
	real_memory_allocated *= 2; // room for complex

	// TODO consider option to add host mem here. For now, just do gpu mem.
	//////	real_values = (float *) fftwf_malloc(sizeof(float) * real_memory_allocated);
	//////	complex_values = (std::complex<float>*) real_values;  // Set the complex_values to point at the newly allocated real values;
//	wxPrintf("\n\n\tAllocating mem\t\n\n");
	cudaErr(cudaMalloc(&real_values_gpu, real_memory_allocated*sizeof(cufftReal)));
	complex_values_gpu = (cufftComplex *)real_values_gpu;
	is_in_memory_gpu = true;

	// Update addresses etc..
	UpdateLoopingAndAddressing(wanted_x_size, wanted_y_size, wanted_z_size);

	if (IsEven(wanted_x_size) == true) padding_jump_value = 2;
	else padding_jump_value = 1;

	// record the full length ( pitch / 4 )
	dims.w = dims.x + padding_jump_value;
	pitch = dims.w * sizeof(float);

	number_of_real_space_pixels = int(dims.x) * int(dims.y) * int(dims.z);
	ft_normalization_factor = 1.0 / sqrtf(float(number_of_real_space_pixels));


	// Set other gpu vals

	is_host_memory_pinned = false;
	is_meta_data_initialized = true;

}

void VectorImage::Deallocate()
{

  if (is_host_memory_pinned)
	{
		cudaErr(cudaHostUnregister(real_values));
		is_host_memory_pinned = false;
	} 
	if (is_in_memory_gpu) 
	{
		cudaErr(cudaFree(real_values_gpu));
		cudaErr(cudaFree(tmpVal));
		cudaErr(cudaFree(tmpValComplex));
		is_in_memory_gpu = false;
	}	

	BufferDestroy();


  if (is_fft_planned)
  {
    cudaErr(cufftDestroy(cuda_plan_inverse));
    cudaErr(cufftDestroy(cuda_plan_forward));
    is_fft_planned = false;
		is_set_complexConjMulLoad = false;
  }

    //  if (is_cublas_loaded)
    //  {
    //    cudaErr(cublasDestroy(cublasHandle));
    //    is_cublas_loaded = false;
    //  }

  if (is_allocated_mask_CSOS)
  {
    mask_CSOS->Deallocate();
		delete mask_CSOS;
  }

  if (is_allocated_image_buffer)
  {
    image_buffer->Deallocate();
		delete image_buffer;
  }

  if (is_allocated_clip_into_mask) 
	{
		cudaErr(cudaFree(clip_into_mask));
		delete clip_into_mask;
	}

}
#pragma endregion protected