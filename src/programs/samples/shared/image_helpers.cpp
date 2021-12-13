void print2DArray(Image &image) {
  int i = 0;
  wxPrintf("Image real space data:\n");
	for (int z = 0; z < image.logical_z_dimension; z++)
	{
	  for (int y = 0; y < image.logical_y_dimension; y++)
	  {
		for (int x = 0; x < image.logical_x_dimension; x++)
		{
      	wxPrintf("%f\t", image.real_values[i]);
		 
		  i++;
		}
    wxPrintf("\n");
		i += image.padding_jump_value;
	  }
    wxPrintf("\n");
	}
}

void PrintArray(float *p, int maxLoops = 10)
{
  wxPrintf("Starting loop through array.\n");

  if (p == nullptr)
  {
	wxPrintf("pointer is null, aborting.\n");
	return;
  }
  for (int i = 0; i < maxLoops; i++)
  {
	wxPrintf("%s \n", std::to_string(i));
	// wxPrintf(" %s\n", *arr);
	// std::cout<< *arr <<" ";
	std::cout << *(p + i) << std::endl;

	p++;
  }
  wxPrintf("Loop done.\n");
}

bool ProperCompareRealValues(Image &first_image, Image &second_image,  float epsilon = 1e-5)
{
  bool passed;
  if (first_image.real_memory_allocated != second_image.real_memory_allocated)
  {

    // wxPrintf(" real_memory_allocated values are not the same. [Failed]\n");
    // wxPrintf(" cpu_image.real_memory_allocated ==  %s\n",
    //         std::to_string(first_image.real_memory_allocated));
    // wxPrintf(" resized_host_image.real_memory_allocated ==  %s\n",
    //         std::to_string(second_image.real_memory_allocated));

    passed = false;
  }
  else
  {

  // print2DArray(first_image);
  // print2DArray(second_image);

	int total_pixels = 0;
	int unequal_pixels = 0;
	// wxPrintf(" real_memory_allocated values are the same. (%s) Starting loop\n", std::to_string(first_image.real_memory_allocated));
	// wxPrintf(" cpu_image.real_values[0] == (%s)\n", std::to_string(first_image.real_values[0]));
	// wxPrintf(" resized_host_image.real_values[0] == (%s)\n", std::to_string(second_image.real_values[0]));

	int i = 0;
	for (int z = 0; z < first_image.logical_z_dimension; z++)
	{
	  for (int y = 0; y < first_image.logical_y_dimension; y++)
	  {
		for (int x = 0; x < first_image.logical_x_dimension; x++)
		{
		  if (std::fabs(first_image.real_values[i] - second_image.real_values[i]) > epsilon) {
            unequal_pixels++;
            if (unequal_pixels < 50) {
              wxPrintf(" Unequal pixels at position: %s, value 1: %s, value 2: %s.\n", std::to_string(i),
                                                                                        std::to_string(first_image.real_values[i]),
                                                                                        std::to_string(second_image.real_values[i]));
            }
              //wxPrintf(" Diff: %f\n", first_image.real_values[i]-second_image.real_values[i]);
        }
		  total_pixels++;
		  i++;
		}
		i += first_image.padding_jump_value;
	  }
	}

	passed = true;
	if (unequal_pixels > 0)
	{
	  int unequal_percent = 100 * (unequal_pixels / total_pixels);

	  wxPrintf("%d out of %d (%*.2lf%%) of pixels are not equal between compared images. [Failed]\n", unequal_pixels, total_pixels, unequal_percent);

	  wxPrintf("Padding values 1: %s, and 2: %s\n",
	           std::to_string(first_image.padding_jump_value),
	           std::to_string(second_image.padding_jump_value));
	  passed = false;
	}
  }

// TODO make this match what is done in disk_io.cpp, or better move the print test to classes.
  if (passed)
  {
	  wxPrintf("\n\tCompared images are ~equal (epsilon= %f).  [Success]\n", epsilon);
  }
  else
  {
	  wxPrintf("\tCompared images are not equal (epsilon= %f). [Failed]\n", epsilon);
  }
  return passed;
}