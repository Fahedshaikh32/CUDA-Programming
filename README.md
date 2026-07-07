<div align="center">

# ⚡ CUDA-Programming

**A growing collection of CUDA C/C++ programs and learning exercises exploring GPU computing and parallel programming with NVIDIA CUDA.**

Built while working through NVIDIA's official tutorials and DLI (Deep Learning Institute) material, using Google Colab's free GPU runtime.

[![Language](https://img.shields.io/badge/language-CUDA%20C%2FC%2B%2B-76B900?logo=nvidia&logoColor=white)](#)
[![Environment](https://img.shields.io/badge/environment-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

[Overview](#overview) •
[Repository Structure](#repository-structure) •
[Topics Covered](#topics-covered) •
[Getting Started](#getting-started) •
[Roadmap](#roadmap)

</div>

---

## Overview

This repository documents my hands-on journey into **GPU-accelerated computing** using CUDA. Each notebook takes a C/C++ program, moves it from the CPU to the GPU step by step, and profiles the performance gain along the way — starting from a single GPU thread and working up to a fully parallel grid of thread blocks.

The exercises are run entirely in **Google Colab**, so no local CUDA toolkit or dedicated GPU hardware is required to follow along.

## Repository Structure

```
.
├── even-easier-cuda/
│   └── An_Even_Easier_Introduction_to_CUDA.ipynb   # CPU → single-thread GPU → parallel GPU walkthrough
├── LICENSE
└── README.md
```

As more exercises are completed, each will get its own folder following the same pattern.

## Topics Covered

| Notebook | Concepts |
|---|---|
| **An Even Easier Introduction to CUDA** | `__global__` kernel functions, host vs. device code, Unified Memory (`cudaMallocManaged`), kernel launch syntax (`<<<blocks, threads>>>`), thread indexing (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`), grid-stride loops, and profiling with `nvprof` |

Broadly, the series builds up through:

1. **Baseline CPU implementation** — a plain C++ array-add function
2. **Single-thread GPU kernel** — introducing `__global__` and Unified Memory
3. **Multi-threaded kernel** — parallelizing across threads in a single block
4. **Multi-block kernel** — scaling across a full grid of thread blocks using grid-stride loops
5. **Profiling** — measuring the speedup at each stage with `nvprof` and `nvidia-smi`

## Getting Started

No local setup is required — everything runs in Google Colab.

1. Open the notebook directly in Colab:
   [`even-easier-cuda/An_Even_Easier_Introduction_to_CUDA.ipynb`](./even-easier-cuda/An_Even_Easier_Introduction_to_CUDA.ipynb)
2. In Colab, make sure a GPU runtime is selected: `Runtime → Change runtime type → GPU`
3. Run the cells top to bottom — each one writes a `.cu`/`.cpp` file, compiles it with `nvcc`/`g++`, and executes it directly from the notebook

### Running locally instead (optional)

If you have the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and a CUDA-capable GPU installed locally:

```bash
nvcc add_grid.cu -o add_grid
./add_grid
nvprof ./add_grid   # profile kernel execution time
```

## Roadmap

- [ ] Add more DLI/CUDA exercises as they're completed (matrix multiplication, reductions, shared memory, streams)
- [ ] Add a benchmarks table comparing CPU vs. GPU timings across notebooks
- [ ] Explore CUDA libraries (cuBLAS, Thrust)
- [ ] Add notes on memory coalescing and occupancy optimization

## License

This project is licensed under the MIT License — see [`LICENSE`](./LICENSE) for details. The introductory notebook is based on NVIDIA's publicly available *["An Even Easier Introduction to CUDA"](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)* blog post and DLI material; all credit for the original tutorial content belongs to NVIDIA.

## Author

**Fahed Shaikh**

<div align="center">

If you find this useful for learning CUDA, consider giving it a ⭐

</div>
