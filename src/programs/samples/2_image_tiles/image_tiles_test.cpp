bool TestImageTiling(wxString image_filename, wxString temp_directory) {
  
    bool passed = true;
    int z_cpu, z_gpu;
    bool print_verbose = true;
    if (print_verbose) wxPrintf("Starting image tiles test.\n");
    MRCFile input_file(image_filename.ToStdString(), false);
    wxString temp_filename = temp_directory + "/tmp1.mrc";

    MRCFile output_file(temp_filename.ToStdString(), false);

    if (print_verbose) wxPrintf("\tReading image to CPU.\n");
    Image cpu_image;
    
    cpu_image.ReadSlice(&input_file, 1);
    //cpu_image.SetToConstant(1.0f);
    //cpu_image.AddConstant(1.0f);

    if (print_verbose) wxPrintf("\tCopying CPU Image.\n");
    Image cpu_copy;
    cpu_copy.ReadSlice(&input_file, 1);
    //cpu_copy.AddConstant(1.0f);

    if (print_verbose) wxPrintf("\tConverting CPU image to TileImage.\n");
  
    TiledImage tiled_image(cpu_image, 3, print_verbose);
   

    if (print_verbose) wxPrintf("\tProcessing TileImage.\n");
    tiled_image.BlankProcess();


    if (print_verbose) wxPrintf("\tComparing original image with tiled image.\n");
    passed = ProperCompareRealValues(cpu_image, cpu_copy);

    if (print_verbose) wxPrintf("Image tiles test done.");
    if (print_verbose) wxPrintf(passed ? "[Success]\n" : "[Failed]");

    // if (print_verbose) wxPrintf(" Exporting images.\n");

    // std::string name1 = std::tmpnam(nullptr);
    // wxPrintf("Tmp outputs are at %s\n", name1.c_str());
    
    // cpu_image.QuickAndDirtyWriteSlice( name1 + "_tiled_and_stiched.mrc", 1, true, 1.0);
    // cpu_copy.QuickAndDirtyWriteSlice( name1 + "_original.mrc", 1, true, 1.0);

    return passed;

}