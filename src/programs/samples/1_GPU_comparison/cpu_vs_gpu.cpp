/*
 * cpu_vs_gpu.hpp
 *
 *  Created on: Aug 10, 2021
 *      Author: B.A. Himes, Shiran Dror
 *
 *      Goal:
 *      	Compare resize functions on CPU and GPU
 *
 *
 */

#include <iostream>
#include <string>
#include <wx/string.h>
#include <wx/wxcrtvararg.h>

bool DoCPUvsGPUResize(wxString hiv_image_80x80x1_filename,  wxString temp_directory, bool print_verbose = false)
{

  bool passed = true;
 
  {
    MRCFile input_file(hiv_image_80x80x1_filename.ToStdString(), false);

    wxString temp_filename = temp_directory + "/tmp1.mrc";

    MRCFile output_file(temp_filename.ToStdString(), false);
    {
      if (print_verbose) wxPrintf("\tStarting CPU vs GPU compare (not resizing).\n");
      Image cpu_image;
      cpu_image.ReadSlice(&input_file, 1);

      Image from_gpu_image;
      if (print_verbose) wxPrintf("\t\tReading image from drive to CPU.\n");
      from_gpu_image.ReadSlice(&input_file, 1);
      if (print_verbose) wxPrintf("\t\tInitiating GPU image.\n");
      GpuImage gpu_image(from_gpu_image);
      if (print_verbose) wxPrintf("\t\tCopying image from CPU to GPU.\n");
      gpu_image.CopyHostToDevice();
      if (print_verbose) wxPrintf("\t\tCopying image from GPU to a new CPU image.\n");
      Image new_cpu_image_from_gpu = gpu_image.CopyDeviceToNewHost(true, true);

 

      if (print_verbose) wxPrintf("\t\tComparing images.\n");
      passed = ProperCompareRealValues(cpu_image, new_cpu_image_from_gpu);
      if (print_verbose) wxPrintf("\tCPU vs GPU compare (not resizing) ended. ");
      if (print_verbose) wxPrintf(passed ? "[Success]\n" : "[Failed]\n");
    }
    //return passed;
      // resize test
    {
      Image cpu_image;
      cpu_image.ReadSlice(&input_file, 1);
      if (print_verbose) wxPrintf("\tStarting CPU resize.\n");
      cpu_image.Resize(40, 40, 1, 0);

      Image host_image;

      host_image.ReadSlice(&input_file, 1);

      GpuImage device_image(host_image);

      if (print_verbose) wxPrintf("\t\tGPU image initiated from host image.\n");
      device_image.CopyHostToDevice();

      if (print_verbose) wxPrintf("\t\tImage copied from host to device.\n");

      device_image.Resize(40, 40, 1, 0);

      if (print_verbose) wxPrintf("\t\tImage resized.\n");

      Image resized_host_image = device_image.CopyDeviceToNewHost(true, true);

      if (print_verbose) wxPrintf("\t\tImage copied from device to host.\n");

      passed = ProperCompareRealValues(cpu_image, resized_host_image);
    }
  }
  if (print_verbose) wxPrintf("\tCPU vs. GPU real space resize test ended. ");
  if (print_verbose) wxPrintf(passed ? "[Success]\n" : "[Failed]\n");
  return passed;
}

bool DoGPUComplexResize(wxString hiv_image_80x80x1_filename, wxString temp_directory, bool print_verbose = false) 
{
  bool passed = true;

  MRCFile input_file(hiv_image_80x80x1_filename.ToStdString(), false);

  wxString temp_filename = temp_directory + "/tmp1.mrc";

  MRCFile output_file(temp_filename.ToStdString(), false);

  if (print_verbose) wxPrintf("\tStarting CPU vs GPU fourier space resizing.\n");
  Image cpu_image;
  cpu_image.ReadSlice(&input_file, 1);

    // resize test

  if (print_verbose) wxPrintf("\t\tTransforming CPU image to Fourier space.\n");
  cpu_image.ForwardFFT();

  if (print_verbose) wxPrintf("\t\tResizing CPU Image.\n");
  cpu_image.Resize(40, 40, 1, 0);
  cpu_image.BackwardFFT();

  Image host_image;

  host_image.ReadSlice(&input_file, 1);

  GpuImage device_image(host_image);

  if (print_verbose) wxPrintf("\t\tGPU image initiated from host image.\n");
  device_image.CopyHostToDevice();
  
  if (print_verbose)wxPrintf("\t\tTransforming to Fourier space.\n");
  device_image.ForwardFFT();


  if (print_verbose) wxPrintf("\t\tResizing image.\n");
  device_image.Resize(40, 40, 1, 0);


  if (print_verbose) wxPrintf("\t\tTransforming GPU image to real space.\n");
  device_image.BackwardFFT();

  if (print_verbose) wxPrintf("\t\tCopying image from device to host.\n");
  Image resized_host_image = device_image.CopyDeviceToNewHost(true, true);

  
  passed = ProperCompareRealValues(cpu_image, resized_host_image);

  

  //if (print_verbose) wxPrintf(" Exporting images.\n");
  // wxPrintf("resized_host_image.is_in.real_space: %d\n", resized_host_image.is_in_real_space);
  // wxPrintf("resized_host_image.is_in.is_in_memory: %d\n", resized_host_image.is_in_memory);

  // std::string name1 = std::tmpnam(nullptr);
  // wxPrintf("Tmp outputs are at %s\n", name1.c_str());
  
  // cpu_image.QuickAndDirtyWriteSlice( name1 + "_cpu.mrc", 1, true, 1.0);
  // resized_host_image.QuickAndDirtyWriteSlice( name1 + "_gpu.mrc", 1, true, 1.0);

  if (print_verbose) wxPrintf("\tCPU vs. GPU fourier space resize test ended. ");
  if (print_verbose) wxPrintf(passed ? "[Success]\n" : "[Failed]\n");
  return passed;

}