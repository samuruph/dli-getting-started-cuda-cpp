#include "dli.h"

int main() {
    thrust::universal_vector<int> vec{ 1, 2, 3 };
    thrust::for_each(thrust::host, vec.begin(), vec.end(), []__host__(int val) {
        std::printf("printing %d on %s\n", val, dli::execution_space());
    });
}
