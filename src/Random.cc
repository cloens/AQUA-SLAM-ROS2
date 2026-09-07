#include "Random.h"

#include <mutex>
#include <random>
#include <stdexcept>

namespace ORB_SLAM3
{
namespace
{
std::mutex randomMutex;
std::mt19937 randomEngine;
bool randomSeeded = false;
}

void SeedRandomOnce(unsigned int seed)
{
    std::lock_guard<std::mutex> lock(randomMutex);
    if (!randomSeeded) {
        randomEngine.seed(seed);
        randomSeeded = true;
    }
}

int RandomInt(int minimum, int maximum)
{
    if (minimum > maximum)
        throw std::invalid_argument("RandomInt requires minimum <= maximum");
    std::lock_guard<std::mutex> lock(randomMutex);
    if (!randomSeeded) {
        randomEngine.seed(0U);
        randomSeeded = true;
    }
    return std::uniform_int_distribution<int>(minimum, maximum)(randomEngine);
}

} // namespace ORB_SLAM3
