## Section 1.3: Extending Algorithms
In this section, we will explore how to extend existing algorithms using custom iterators in CUDA programming. **Iterators** are objects that allow you to traverse a container (like an array or vector) without exposing the underlying representation of the container. By creating custom iterators, you can define new ways to access and manipulate data, enabling you to extend the functionality of existing algorithms.

### Contents
- [Extended algorithm example](#extended-algorithm-example)
- [Iterators](#iterators)
  - [Simple counting iterator](#simple-counting-iterator)
  - [Simple transform iterator](#simple-transform-iterator)
  - [Simple zip iterator](#simple-zip-iterator)
  - [Combining input iterators](#combining-input-iterators)
  - [Transforming output iterator](#transforming-output-iterator)
  - [Discard iterator](#discard-iterator)
  - [CUDA fancy iterators](#cuda-fancy-iterators)
- [Performance comparison](#performance-comparison)
- [Exercise: Computing variance](#exercise-computing-variance)
- [Summary](#summary)

### Extended algorithm example
Let's say that we have to find the maximum change in temperature made in the current step of our cooling simulation. Here the code that does that:
```cpp
#include "dli.h"

float naive_max_change(const thrust::universal_vector<float>& a, 
                       const thrust::universal_vector<float>& b) 
{
    // allocate vector to store `a - b`
    thrust::universal_vector<float> unnecessarily_materialized_diff(a.size());

    // compute products
    thrust::transform(thrust::device, 
                      a.begin(), a.end(),                       // first input sequence
                      b.begin(),                                // second input sequence
                      unnecessarily_materialized_diff.begin(),  // result
                      []__host__ __device__(float x, float y) { // transformation (abs diff)
                         return abs(x - y); 
                      });

    // compute max difference
    return thrust::reduce(thrust::device, 
                          unnecessarily_materialized_diff.begin(), 
                          unnecessarily_materialized_diff.end(), 
                          0.0f, thrust::maximum<float>{});
}

int main() 
{
    float k = 0.5;
    float ambient_temp = 20;
    thrust::universal_vector<float> temp[] = {{ 42, 24, 50 }, { 0, 0, 0}};
    auto transformation = [=] __host__ __device__ (float temp) { return temp + k * (ambient_temp - temp); };

    std::printf("step  max-change\n");
    for (int step = 0; step < 3; step++) {
        thrust::universal_vector<float> &current = temp[step % 2];
        thrust::universal_vector<float> &next = temp[(step + 1) % 2];

        thrust::transform(thrust::device, current.begin(), current.end(), next.begin(), transformation);
        std::printf("%d     %.2f\n", step, naive_max_change(current, next));
    }
}
```
You can use:
- `thrust::transform` to compute the absolute difference between the current and next temperature arrays.
- `thrust::reduce` to find the maximum value in the difference array.

In this code, there are several issues that can be optimized:
1. **Unnecessary materialization**: The code creates an intermediate vector `unnecessarily_materialized_diff` to store the absolute differences between the two temperature arrays. This vector consumes additional memory and introduces extra data transfer overhead between the host and device.
2. **Multiple kernel launches**: The code launches two separate kernels: one for the transformation and another for the reduction. Each kernel launch incurs overhead, which can impact performance, especially for small data sizes.
    - `thrust::transform` launches a kernel to compute the absolute differences between the two arrays.
        ```cpp
        thrust::transform(thrust::device, 
                        a.begin(), a.end(),                       // first input sequence
                        b.begin(),                                // second input sequence
                        diff.begin(),                             // result
                        []__host__ __device__(float x, float y) { // transformation (abs diff)
                            return abs(x - y); 
                        });
        ```
    - `thrust::reduce` with `thrust::maximum<float>{}` launches another kernel to find the maximum value in the difference array.
        ```cpp
        return thrust::reduce(thrust::device, 
                            unnecessarily_materialized_diff.begin(), 
                            unnecessarily_materialized_diff.end(), 
                            0.0f, thrust::maximum<float>{});
        ```
3. **Memory (this point is related to the second one)**: We have a total of `4 * N` memory accesses:
    - `thrust::transform` reads `2 * N` elements from global memory (the two input arrays a and b) and writes `N` elements to global memory (the difference array).
    - `thrust::reduce` reads `N` elements from global memory (the difference array) to compute the maximum value.

We could improve this by using a single for loop that computes the absolute differences and keeps track of the maximum change in a single pass, avoiding the need for an intermediate array and reducing the number of kernel launches:
```cpp
float max_diff = 0;
for (int i = 0; i < a.size(); i++) {
  max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
}
```
This version only requires `2 * N` memory accesses (reading from both input arrays) and eliminates the need for an intermediate array, thus improving both memory usage and performance. This will lead to a 2x speedup compared to the previous implementation.

### Iterators
Luckily, there is a way to avoid materializing the intermediate difference array while still using `thrust::transform` and `thrust::reduce`. This can be achieved by creating a **custom iterator** that computes the absolute difference on-the-fly as we iterate over the two input arrays. 

**Iterators** are objects that allow you to traverse a container (like an array or vector) without exposing the underlying representation of the container. They provide a way to access elements in a sequence one at a time. They can be seen as generalized pointers that can be incremented, dereferenced, and compared:
- A pointer, `int* p`, points to a sequence of integers in memory.
- You can deference a pointer to access the value it points to: `*p`.
- You can increment the pointer to point to the next integer in the sequence: `p++`.
- You can compare two pointers to check if they point to the same location: `p1 == p2`.

The following code shows how to use a pointer as an iterator:
```cpp
#include "dli.h"

int main() 
{
    std::array<int, 3> a{ 0, 1, 2 };

    int *pointer = a.data();

    std::printf("pointer[0]: %d\n", pointer[0]); // prints 0
    std::printf("pointer[1]: %d\n", pointer[1]); // prints 1
}
```

The operators can be overloaded to create custom iterator types that provide specific behaviors when accessing elements. This allows you to create iterators that can perform transformations, combine multiple sequences, or even generate values on-the-fly without storing them in memory.

### Simple counting iterator
C++ allows operator overloading, so you can create your own iterator types by defining a class that implements the necessary operators (like `[]`, `++`, `*`, and `==`). You can simply use a struct with an overloaded `operator[]` to create a simple counting iterator for example. This will **overload the subscript operator `[]`** to the specific behavior you want when accessing elements through the iterator.

> **Note**: You can overload a specific operator by defining a member function in your class with the name `operator<operator_symbol>`, where `<operator_symbol>` is the symbol of the operator you want to overload (e.g., `+`, `-`, `[]`, etc.).

Here is an example of a simple counting iterator that generates a sequence of integers starting from a given value:
```cpp
#include "dli.h"

struct counting_iterator 
{
  int operator[](int i) 
  {
    return i;
  }
};

int main() 
{
  counting_iterator it;

  std::printf("it[0]: %d\n", it[0]); // prints 0
  std::printf("it[1]: %d\n", it[1]); // prints 1
}
```

In this example, the `counting_iterator` struct defines an `operator[]` that takes an index `i` and returns the value `i`. This allows you to use the iterator like an array, where `it[0]` returns `0`, `it[1]` returns `1`, and so on, without storing any values in memory.


### Simple transform iterator
You can create a **transform iterator** that applies a transformation to the values it accesses. For example, you can create a transform iterator that multiplies each accessed value by 2:
```cpp
#include "dli.h"

struct transform_iterator 
{
  int *a;

  int operator[](int i) 
  {
    return a[i] * 2;
  }
};

int main() 
{
  std::array<int, 3> a{ 0, 1, 2 };

  transform_iterator it{a.data()};

  std::printf("it[0]: %d\n", it[0]); // prints 0 (0 * 2)
  std::printf("it[1]: %d\n", it[1]); // prints 2 (1 * 2)
}
```

### Simple zip iterator
An important type of iterator is the **zip iterator**, that allows you to combine multiple input sequences into a single iterator. For example, you can create a zip iterator that combines two arrays and allows you to access pairs of elements from both arrays:
```cpp
#include "dli.h"

struct zip_iterator 
{
  int *a;
  int *b;

  std::tuple<int, int> operator[](int i) 
  {
    return {a[i], b[i]};
  }
};

int main() 
{
  std::array<int, 3> a{ 0, 1, 2 };
  std::array<int, 3> b{ 5, 4, 2 };

  zip_iterator it{a.data(), b.data()};

  std::printf("it[0]: (%d, %d)\n", std::get<0>(it[0]), std::get<1>(it[0])); // prints (0, 5)
  std::printf("it[0]: (%d, %d)\n", std::get<0>(it[1]), std::get<1>(it[1])); // prints (1, 4)
}
```
In this example, the `zip_iterator` struct defines an `operator[]` that takes an index `i` and returns a tuple containing the `i`-th elements from both arrays `a` and `b`. This allows you to access pairs of elements from both arrays using the zip iterator and the `std::get` function to extract the individual elements from the tuple.

### Combining input iterators
One powerful use case of iterators is to combine them to create new iterators that perform complex transformations. For example, you can combine a zip iterator and a transform iterator to create an iterator that computes the absolute difference between corresponding elements of two arrays:
```cpp
#include "dli.h"

struct zip_iterator 
{
  int *a;
  int *b;

  std::tuple<int, int> operator[](int i) 
  {
    return {a[i], b[i]};
  }
};

struct transform_iterator 
{
  zip_iterator zip;

  int operator[](int i) 
  {
    auto [a, b] = zip[i];
    return abs(a - b);
  }
};

int main() 
{
  std::array<int, 3> a{ 0, 1, 2 };
  std::array<int, 3> b{ 5, 4, 2 };

  zip_iterator zip{a.data(), b.data()};
  transform_iterator it{zip};

  std::printf("it[0]: %d\n", it[0]); // prints 5
  std::printf("it[0]: %d\n", it[1]); // prints 3
}
```
In this example, the `transform_iterator` struct combines a `zip_iterator` to access pairs of elements from two arrays and computes the absolute difference between them in its `operator[]`. This allows you to create a new iterator that directly provides the absolute differences without needing to create an intermediate array to store the results.

### Transforming output iterator
You can also create output iterators that transform values as they are written to a container. For example, you can create an output iterator that divides each value by 2 before storing it in an array:
```cpp
#include "dli.h"

struct wrapper
{
   int *ptr; 

   void operator=(int value) {
      *ptr = value / 2;
   }
};

struct transform_output_iterator 
{
  int *a;

  wrapper operator[](int i) 
  {
    return {a + i};
  }
};

int main() 
{
  std::array<int, 3> a{ 0, 1, 2 };
  transform_output_iterator it{a.data()};

  it[0] = 10;
  it[1] = 20;

  std::printf("a[0]: %d\n", a[0]); // prints 5
  std::printf("a[1]: %d\n", a[1]); // prints 10
}
```
In this example, the `transform_output_iterator` struct defines an `operator[]` that returns a `wrapper` object. The `wrapper` struct overloads the assignment operator `=` to divide the assigned value by 2 before storing it in the array. This allows you to use the output iterator to write transformed values directly to the array.

So, the output will be:
```
a[0]: 5
a[1]: 10
```
because the values `10` and `20` were divided by `2` before being stored in the array.

### Discard iterator
You can create a **discard iterator** that ignores any values written to it. This can be useful when you want to perform computations without storing the results. Here is an example of a discard iterator:
```cpp
#include "dli.h"

struct wrapper
{
   void operator=(int value) {
      // discard value
   }
};

struct discard_iterator 
{
  wrapper operator[](int i) 
  {
    return {};
  }
};

int main() 
{
  discard_iterator it{};

  it[0] = 10;
  it[1] = 20;
}
```
In this example, the `discard_iterator` struct defines an `operator[]` that returns a `wrapper` object. The `wrapper` struct overloads the assignment operator `=` to simply ignore any assigned value. This allows you to use the discard iterator to perform computations without storing the results.

#### CUDA fancy iterators
The CUDA Thrust library provides several **fancy iterators** that can be used to create complex data access patterns without the need for intermediate storage. Some of the most commonly used fancy iterators in Thrust include:
- `thrust::transform_iterator`: Applies a transformation function to each element as it is accessed.
- `thrust::zip_iterator`: Combines multiple input sequences into a single iterator that provides tuples of elements.
- `thrust::permutation_iterator`: Accesses elements in a sequence based on a set of indices.
- `thrust::counting_iterator`: Generates a sequence of integers on-the-fly.

In our example of computing the maximum change in temperature, we can use a combination of `thrust::zip_iterator` and `thrust::transform_iterator` to create an iterator that computes the absolute differences between the two temperature arrays on-the-fly. This allows us to avoid materializing the intermediate difference array and reduces memory usage and kernel launches.
Here we could use:
- `thrust::zip_iterator` to combine the two input temperature arrays into a single iterator that provides pairs of elements. Here an example:
    ```cpp
    #include "dli.h"

    int main() 
    {
        // allocate and initialize input vectors
        thrust::universal_vector<float> a{ 31, 22, 35 };
        thrust::universal_vector<float> b{ 25, 21, 27 };

        // zip two vectors into a single iterator
        auto zip = thrust::make_zip_iterator(a.begin(), b.begin());

        thrust::tuple<float, float> first = *zip;
        std::printf("first: (%g, %g)\n", thrust::get<0>(first), thrust::get<1>(first));

        zip++;

        thrust::tuple<float, float> second = *zip;
        std::printf("second: (%g, %g)\n", thrust::get<0>(second), thrust::get<1>(second));
    }
    ```
    `*zip` is an iterator that combines the two input vectors `a` and `b`. Dereferencing `*zip` gives you a tuple containing the first elements of both vectors, and incrementing `zip` moves to the next pair of elements. We can access the individual elements of the tuple using `thrust::get<index>(tuple)`.
- `thrust::transform_iterator` to apply a transformation function that computes the absolute difference between the two temperature arrays. Here is an example:
    ```cpp
    #include "dli.h"

    int main() 
    {
        thrust::universal_vector<float> a{ 31, 22, 35 };
        thrust::universal_vector<float> b{ 25, 21, 27 };

        auto zip = thrust::make_zip_iterator(a.begin(), b.begin());
        auto transform = thrust::make_transform_iterator(zip, []__host__ __device__(thrust::tuple<float, float> t) {
            return abs(thrust::get<0>(t) - thrust::get<1>(t));
        });

        std::printf("first: %g\n", *transform); // absolute difference of `a[0] = 31` and `b[0] = 25`

        transform++;

        std::printf("second: %g\n", *transform); // absolute difference of `a[1] = 22` and `b[1] = 21`
    }
    ```
    In this example, the `transform` iterator applies a lambda function that computes the absolute difference between the two elements in the tuple provided by the `zip` iterator. Dereferencing `*transform` gives you the absolute difference for the current pair of elements, and incrementing `transform` moves to the next absolute difference.
- Finally, we can use `thrust::reduce` with the `transform` iterator to compute the maximum absolute difference without materializing the intermediate array:
    ```cpp
    float max_change(const thrust::universal_vector<float>& a, 
                 const thrust::universal_vector<float>& b) 
    {
        auto zip = thrust::make_zip_iterator(a.begin(), b.begin());
        auto transform = thrust::make_transform_iterator(zip, []__host__ __device__(thrust::tuple<float, float> t) {
            return abs(thrust::get<0>(t) - thrust::get<1>(t));
        });

        // compute max difference
        return thrust::reduce(thrust::device, transform, transform + a.size(), 0.0f, thrust::maximum<float>{});
    }
    ```
    In this code, `transform` is the beginning of the iterator that computes the absolute differences on-the-fly, and `transform + a.size()` is the end of the iterator. The `thrust::reduce` function computes the maximum value from the absolute differences without needing to store them in an intermediate array.

The overall code will look like this:
```cpp
#include "dli.h"

float max_change(const thrust::universal_vector<float>& a, 
                 const thrust::universal_vector<float>& b) 
{
    auto zip = thrust::make_zip_iterator(a.begin(), b.begin());
    auto transform = thrust::make_transform_iterator(zip, []__host__ __device__(thrust::tuple<float, float> t) {
        return abs(thrust::get<0>(t) - thrust::get<1>(t));
    });

    // compute max difference
    return thrust::reduce(thrust::device, transform, transform + a.size(), 0.0f, thrust::maximum<float>{});
}

int main() 
{
    float k = 0.5;
    float ambient_temp = 20;
    thrust::universal_vector<float> temp[] = {{ 42, 24, 50 }, { 0, 0, 0}};
    auto transformation = [=] __host__ __device__ (float temp) { return temp + k * (ambient_temp - temp); };

    std::printf("step  max-change\n");
    for (int step = 0; step < 3; step++) {
        thrust::universal_vector<float> &current = temp[step % 2];
        thrust::universal_vector<float> &next = temp[(step + 1) % 2];

        thrust::transform(thrust::device, current.begin(), current.end(), next.begin(), transformation);
        std::printf("%d     %.2f\n", step, max_change(current, next));
    }
}
```
and can be compiled and run with:
```bash
nvcc --extended-lambda -o /tmp/a.out Sources/optimized-max-diff.cu # build executable
/tmp/a.out                                                  # run executable
```

### Performance comparison
We can compare the performance of the naive and optimized implementations of the `max_change` function by measuring the execution time for both versions. Here is an example of how to do this:
```cpp
#include "dli.h"

float naive_max_change(const thrust::universal_vector<float>& a, 
                       const thrust::universal_vector<float>& b) 
{
    thrust::universal_vector<float> diff(a.size());
    thrust::transform(thrust::device, a.begin(), a.end(), b.begin(), diff.begin(),
                      []__host__ __device__(float x, float y) {
                         return abs(x - y); 
                      });
    return thrust::reduce(thrust::device, diff.begin(), diff.end(), 0.0f, thrust::maximum<float>{});
}

float max_change(const thrust::universal_vector<float>& a, 
                 const thrust::universal_vector<float>& b) 
{
    auto zip = thrust::make_zip_iterator(a.begin(), b.begin());
    auto transform = thrust::make_transform_iterator(zip, []__host__ __device__(thrust::tuple<float, float> t) {
        return abs(thrust::get<0>(t) - thrust::get<1>(t));
    });
    return thrust::reduce(thrust::device, transform, transform + a.size(), 0.0f, thrust::maximum<float>{});
}

int main() 
{
    // allocate vectors containing 2^28 elements
    thrust::universal_vector<float> a(1 << 28);
    thrust::universal_vector<float> b(1 << 28);

    thrust::sequence(a.begin(), a.end());
    thrust::sequence(b.rbegin(), b.rend());

    auto start_naive = std::chrono::high_resolution_clock::now();
    naive_max_change(a, b);
    auto end_naive = std::chrono::high_resolution_clock::now();
    const double naive_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_naive - start_naive).count();

    auto start = std::chrono::high_resolution_clock::now();
    max_change(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    const double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::printf("iterators are %g times faster than naive approach\n", naive_duration / duration);
}
```
It will print:
```
iterators are 54.4286 times faster than naive approach
```
, showing a significant performance improvement by using fancy iterators to avoid unnecessary materialization and reduce memory accesses.

### Exercise: Computing variance
In this exercise, you will implement a function to compute the variance of a set of temperature readings using Thrust fancy iterators. The variance is defined as the average of the squared differences from the mean.

The starting point of the solution is this:
```cpp
#include "dli.h"

float variance(const thrust::universal_vector<float> &x, float mean) {
  // update the following line so that dereferencing `squared_differences`
  // returns `(xi - mean) * (xi - mean)`
  auto squared_differences = ...;

  return thrust::reduce(thrust::device, squared_differences,
                        squared_differences + x.size()) /
         x.size();
}

float mean(thrust::universal_vector<float> vec) {
  return thrust::reduce(thrust::device, vec.begin(), vec.end()) / vec.size();
}

int main() {
  float ambient_temp = 20;
  thrust::universal_vector<float> prev{42, 24, 50};
  thrust::universal_vector<float> next{0, 0, 0};

  std::printf("step  variance\n");
  for (int step = 0; step < 3; step++) {
    thrust::transform(thrust::device, prev.begin(), prev.end(), next.begin(),
                      [=] __host__ __device__(float temp) {
                        return temp + 0.5 * (ambient_temp - temp);
                      });
    std::printf("%d     %.2f\n", step, variance(next, mean(next)));
    next.swap(prev);
  }
}
```

The correct solution would be:
```cpp
auto squared_differences = thrust::make_transform_iterator(
  x.begin(), [mean] __host__ __device__(float value) {
    return (value - mean) * (value - mean);
  });
```

Basically, we would:
1. Use `thrust::make_transform_iterator` to create a transform iterator that applies a lambda function to each element of the input vector `x` (by passing `x.begin()` as the first argument).
2. The lambda function takes a single float value (an element from the vector `x`) and computes the squared difference from the mean: `(value - mean) * (value - mean)`.
3. The lambda function is annotated with `__host__ __device__` to ensure it can be executed on both the host and device.

Overall, this transform iterator will compute the squared differences on-the-fly as we iterate over the input vector `x`, allowing us to compute the variance without needing to create an intermediate array to store the squared differences.

> **Note 1**: When using `thrust::reduce`, we need to pass `squared_differences` as a pointer to the beginning of the iterator range and `squared_differences + x.size()` as a pointer to the end of the iterator range. This is because `thrust::reduce` expects iterators that define a range of elements to operate on.

> **Note 2:** Do not pass `*squared_differences` to `thrust::reduce`, as this would dereference the iterator and pass a single value instead of the entire range of elements.

### Summary
In this section, we explored:
- The concept of **iterators** and how they allow you to traverse containers without exposing their underlying representation.
- How to create **custom iterators** by overloading operators to define specific behaviors when accessing elements.
- The use of **Thrust fancy iterators** like `thrust::transform_iterator` and `thrust::zip_iterator` to create complex data access patterns without intermediate storage.
- How to optimize algorithms by combining iterators to perform transformations on-the-fly, reducing memory usage and kernel launches.
- An exercise to compute variance using Thrust fancy iterators.