#include <Student.1.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <utils/ChronoGPU.hpp>

namespace
{
// TODO: add your functor here

//fill_shared_memory
//scan
//apply_filter
//update_histo

} // namespace

float student1(
    thrust::device_vector<uchar3> &d_input,
    thrust::device_vector<uchar> &d_V)
{
    ChronoGPU chr;
    chr.start();
    // TODO: add your implementation here
    chr.stop();
    return chr.elapsedTime();
}