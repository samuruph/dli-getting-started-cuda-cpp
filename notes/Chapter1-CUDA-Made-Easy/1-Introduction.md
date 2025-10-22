# Chapter 1: CUDA made easy: Accelerating applications with parallel algorithms
Welcome to Chapter 1 of the "Getting Started with CUDA C++" course! In this chapter, you will learn how to accelerate your C++ applications using CUDA by leveraging parallel algorithms and execution models. The chapter is designed to provide a hands-on introduction to CUDA programming concepts, focusing on practical applications and code examples.

## 🎯 Objectives
By the end of this chapter, you will be able to:
- Understand the differences between **parallel** and **serial** execution.
- Control where your C++ code runs on **host** (CPU) and **device** (GPU) using **execution spaces**.
- Control where your data is stored using **memory spaces**, such as **host memory** and **device memory**.
- Refactor standard C++ algorithms to run on the GPU using CUDA.
- Leverage powerful parallel algorithms to accelerate your applications with minimal code changes.
- Use **fancy iterators** to manipulate data in different memory spaces seamlessly.


---

## 📓 Contents
- [Section 1.1: Introduction](1-Introduction.md)
- [Section 1.2: Execution Spaces](2-Execution-Spaces.md)
- [Section 1.3: Extending Algorithms](3-Extending-Algorithms.md)
- [Section 1.4: Vocabulary Types](4-Vocabulary-Types.md)
- [Section 1.5: Serial vs Parallel](5-Serial-vs-Parallel.md)
- [Section 1.6: Memory Spaces](6-Memory-Spaces.md)
- [Section 1.7: Summary](7-Summary.md)

---

## Section 1.1: Introduction
In this section, we have just provided an overview of the chapter objectives and contents. As we progress through the chapter, we will delve into each of these topics in detail, providing code examples and exercises to help you solidify your understanding of CUDA programming concepts.

### Why GPU is better than CPU for parallel computing?
GPUs are designed with a large number of smaller, efficient cores that can handle multiple tasks simultaneously, making them ideal for parallel computing. In contrast, CPUs have fewer cores optimized for sequential serial processing. This architectural difference allows GPUs to excel in tasks that can be parallelized, such as graphics rendering and scientific computations.

Please continue to the next section to learn about execution spaces in CUDA programming!

Have fun coding with CUDA! 🚀