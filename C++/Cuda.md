Resources from https://github.com/Infatoshi/cuda-course
Thanks the video on Youtube https://www.youtube.com/watch?v=86FAWCzIe_4

# Cuda

## Keywords

### Kernel

A kernel is a special function that runs on your GPU (Graphics Card) instead of your CPU. Think of it like giving instructions to a large team of workers (GPU threads) who can all work at the same time. You mark a kernel with __global__ keyword, and it can only return void. Example:

```cpp
__global__ void addNumbers(int *a, int *b, int *result) {
    *result = *a + *b;
}
```

### Grid
The grid represents the entire set of threads launched for a single kernel invocation. Think of it as the overall execution space. It's a collection of thread blocks.

When you launch a kernel, you specify the dimensions of the grid, essentially defining how many blocks will be created. It can be 1D, 2D, or 3D (like a line, sheet, or cube of blocks). It is used for organizing really large computations

- Example: When processing a large image, each block might handle one section of the image

### Block
A block is a group of threads that can cooperate and share data quickly through shared memory.It also can be 1D, 2D, or 3D

- Threads within a block can:

    - Share memory
    - Synchronize with each other
    - Cooperate on tasks
- Example: If processing an image, a block might handle a 16x16 pixel region

### Threads
The thread is the smallest unit of execution in CUDA. Each thread executes the kernel code independently. Within a block, threads are identified by a unique thread ID. This ID allows you to access specific data or perform different operations based on the thread's position within the block.

Threads have their own unique ID to know which data to work on

> Grid > Block > Thread

### Understanding CUDA Thread Indexing

In CUDA, Each thread has a unique identifier that can be used to determine its position within the grid and block. The following variables are commonly used for this purpose:

1. **`threadIdx`**:  
   - A 3-component vector (`threadIdx.x`, `threadIdx.y`, `threadIdx.z`) that gives the thread's position within its block.
   - Example: If you have a 1D block of 256 threads, `threadIdx.x` ranges from `0` to `255`.

2. **`blockDim`**:  
   - A 3-component vector (`blockDim.x`, `blockDim.y`, `blockDim.z`) that specifies the dimensions of the block.
   - Example: If your block is 256 threads in the x-direction, `blockDim.x` is `256`.

3. **`blockIdx`**:  
   - A 3-component vector (`blockIdx.x`, `blockIdx.y`, `blockIdx.z`) that gives the block's position within the grid.
   - Example: If you have a 1D grid of 10 blocks, `blockIdx.x` ranges from `0` to `9`.

4. **`gridDim`**:  
   - A 3-component vector (`gridDim.x`, `gridDim.y`, `gridDim.z`) that specifies the dimensions of the grid.
   - Example: If your grid has 10 blocks in the x-direction, `gridDim.x` is `10`.

### Calculating Global Thread ID

To compute a unique global thread ID (e.g., for accessing elements in a 1D array), you can use the following formula:

```cpp
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx.x * blockDim.x` gives the starting index of the current block.
- `threadIdx.x` gives the thread's position within the block.

## Helper Types and Functions

#### dim3
- A simple way to specify 3D dimensions
- Used for grid and block sizes
- Example:
```cpp
dim3 blockSize(16, 16, 1);  // 16x16x1 threads per block
dim3 gridSize(8, 8, 1);     // 8x8x1 blocks in grid
```

#### <<<>>>

This is used to configure and launch kernels on the GPU. It defines the grid and block dimensions, shared memory size, and stream for kernel execution. Properly configuring these parameters is essential for efficient GPU programming and maximizing performance.

- Special brackets for launching kernels
- Specifies grid and block dimensions
- Example:

```cpp
addNumbers<<<gridSize, blockSize>>>(a, b, result);
```

here `addNumbers` is the name of our kernel and `gridSize` and `blockSize` is the size of grid and block and are `a`, `b`, `result` are the arguments passed to the kernel
