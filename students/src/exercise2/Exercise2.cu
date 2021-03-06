#pragma warning( disable : 4244 ) 

#include <iostream>
#include <exercise2/Exercise2.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/equal.h>


namespace {
        
}

// ==========================================================================================
void Exercise2::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << " -i=<image.ppm> [-o=<image_output_basename.ppm>] [-d=filter_size]"<< std::endl
        << "\twhere <image_input.ppm> is the input image," << std::endl
        << "\t<image_output_basename.ppm> is the basename of the output images,"<<std::endl
        << "\tfilter_size is the size of the median filter." << std::endl
        << std::endl;
}

// ==========================================================================================
void Exercise2::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void Exercise2::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
Exercise2& Exercise2::parseCommandLine(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "i") ) {
        char *value;
        getCmdLineArgumentString(argc, argv, "i", &value);
        std::cout << "Input file is " << value << std::endl;
        inputFileName = std::string(value);
    }
    else
        usageAndExit(argv[0], -1); 
    if( checkCmdLineFlag(argc, argv, "o") ) {
        char*value;
        getCmdLineArgumentString(argc, argv, "o", &value);
        std::cout << "Output file is " << value << std::endl;
        outputFileName_rgb = std::string(value);
    }
    else {
        outputFileName_rgb = inputFileName;
    }
    if( checkCmdLineFlag(argc, argv, "-d") ) {
        int value = getCmdLineArgumentInt(argc, argv, "-d");
        if( value > 16 && value < 1024 ) {
            filter_size = value;
            std::cout << "Found -d=" << value << std::endl;
        }
    }
    reinterpret_cast<StudentWork2*>(student)->parseCommandLine(argc, argv);     
    return *this;
}

void Exercise2::loadImage() 
{
    input = new PPMBitmap(inputFileName.c_str());
    const unsigned size = input->getWidth()*input->getHeight();
    uchar3*const ptr = reinterpret_cast<uchar3*>( input->getPtr() );
    thrust::host_vector<uchar3> h_RGB( ptr, ptr+size);
    d_RGB_in = h_RGB;
    d_V.resize(size);
    d_V_median.resize(size);
    d_RGB_filtered.resize( d_RGB_in.size());
    d_RGB_out.resize( d_RGB_in.size());
}

void Exercise2::run(const bool verbose) {    
    if( verbose )
        std::cout << std::endl << "Convert the image, build the histogram" << std::endl;
    // build a host vector containing the reference
    loadImage();
    ChronoGPU chr;
    StudentWork2& worker = *reinterpret_cast<StudentWork2*>(student);
    chr.start();
    worker.rgb2h( d_RGB_in, d_V );
    worker.median( d_V, d_V_median, input->getWidth(), input->getHeight(), filter_size );
    worker.apply_filter( d_RGB_in, d_V_median, d_RGB_filtered );
    worker.justDoIt( d_RGB_filtered, d_RGB_out, input->getWidth(), input->getHeight() );
    chr.stop();
    if( verbose )
        std::cout << "\t-> Student's Work Done in " << chr.elapsedTime() << " ms" << std::endl;
    saveImage(outputFileName_rgb.c_str(), d_RGB_out, input->getWidth(), input->getHeight());
}

bool Exercise2::check() {
    return true;
}



void Exercise2::saveImage(
    const char*filename, 
    const thrust::host_vector<uchar3>&h_image, 
    const unsigned width, 
    const unsigned height
) {
    PPMBitmap output(input->getWidth(), input->getHeight());
    thrust::copy(h_image.begin(), h_image.end(), reinterpret_cast<uchar3*>(output.getPtr()));
    output.saveTo(filename);
    std::cout << "Image saved to " << filename << std::endl;
}
