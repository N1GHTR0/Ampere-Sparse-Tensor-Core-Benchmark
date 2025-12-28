# Performance Analysis of Sparse Tensor Cores on NVIDIA Ampere Architecture: A Micro-Benchmark Study

![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg) ![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Ampere%20(SM86)-76b900.svg) ![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 🎓 CENG415 Graduation Project
**Author:** Ahmet  
**Institution:** İzmir Institute of Technology (IZTECH)  
**Department:** Computer Engineering

---

## 📖 Project Abstract
This project conducts a rigorous performance analysis of **Sparse Tensor Cores (`mma.sp`)** on NVIDIA Ampere architecture, focusing on **Micro-GEMM** workloads ($16\times8\times32$ to $32\times32\times32$).

Unlike standard benchmarks that rely on Host-side timing (which often includes ~80% driver overhead), this study implements **in-kernel cycle counting (`clock64`)** to isolate the true hardware latency. The research investigates the trade-offs between relying on the **L1 Cache (Global Memory)** versus building explicit **Shared Memory Pipelines** with and without the hardware-accelerated **`ldmatrix`** instruction.

## 🎯 Research Objectives
We evaluate three memory access strategies to determine the "crossover point" where architectural complexity yields performance gains:

1.  **Baseline (Global Memory):** Can the Ampere L1 Cache handle Tensor Core data feeding efficiently without manual software caching?
2.  **Manual Caching (Shared Memory):** Does explicitly loading data into Shared Memory improve performance, or does the setup overhead create a bottleneck?
3.  **Hardware Acceleration (LDMATRIX):** How does the `ldmatrix.sync.aligned` instruction compare to standard loads when data volume increases?

---

## 🔬 Critical Methodology: The "Launch Overhead" Discovery
A key finding of this project is the dominance of **Kernel Launch Overhead** in micro-benchmarks.

* **The Problem:** For small kernels, the CPU-to-GPU launch latency (~1700 ns) is 5x larger than the actual execution time (~300 ns).
* **The Solution:** We used `clock64()` inside the kernels to measure **pure hardware cycles**, filtering out the OS/Driver noise. All results below reflect this isolated hardware performance.

---

## 📊 Benchmark Analysis & Results

### 📉 Scenario 1: Small Workload (One Warp)
*Configuration: 1 Warp (32 Threads) | Matrix: $16\times8\times32$*

**Results (Hardware Cycles - Lower is Better):**
* **Global Memory:** ~465 Cycles 🏆
* **Shared Memory + `ldmatrix`:** ~470 Cycles
* **Manual Shared Memory:** ~502 Cycles

![One Warp Result](OneWarpSparseMatMulCompare.png)

#### 💡 Analysis: The "L1 Efficiency" Zone
In this smallest workload, **Global Memory is the winner.**
* **L1 Cache Dominance:** The matrix sizes are tiny enough to fit perfectly into the L1 Cache. Accessing data from L1 is incredibly fast and requires **zero software overhead**.
* **The Cost of Shared Memory:** Moving data explicitly to Shared Memory involves a "tax": reading from Global, writing to Shared (`STS`), synchronizing (`BAR`), and reading back (`LDS`). For such a small task, this setup cost outweighs the benefits.
* **Manual Shared Memory:** Performs the worst because it pays the "setup tax" but lacks the hardware acceleration of `ldmatrix`.

---

### 📉 Scenario 2: Transition Zone (Four Warps)
*Configuration: 4 Warps (128 Threads) | Matrix: $32\times16\times32$*

**Results (Hardware Cycles):**
* **Global Memory:** ~554 Cycles
* **Shared Memory + `ldmatrix`:** ~555 Cycles
* **Manual Shared Memory:** ~600 Cycles

![Four Warp Result](FourWarpSparseMatMulCompare.png)

#### 💡 Analysis: The Equilibrium
This scenario represents the **tipping point**.
* **Global Memory Still Leads (Barely):** The L1 Cache is still very effective. The data volume hasn't yet reached a level where cache contention becomes a major issue.
* **LDMATRIX Catch-up:** While moving data to Shared Memory is still costly, the `ldmatrix` instruction starts to pay off. Unlike manual loads, `ldmatrix` moves data from Shared Memory to Registers in **128-bit aligned vector chunks**, reducing the number of instructions and bank conflicts. This efficiency helps it catch up to the "free" L1 Cache access.

---

### 🚀 Scenario 3: The Crossover (Eight Warps)
*Configuration: 8 Warps (256 Threads) | Matrix: $32\times32\times32$*

**Results (Hardware Cycles):**
* **Shared Memory + `ldmatrix`:** ~786 Cycles 🏆
* **Global Memory:** ~904 Cycles
* **Manual Shared Memory:** ~978 Cycles

![Eight Warp Result](EightWarpSparseMatMulCompare.png)

#### 💡 Analysis: Hardware Acceleration Wins
Here, the leadership changes hands. **`ldmatrix` becomes the clear winner (~15% faster).**
* **Why Global Memory Fell Behind:** Even though the data *can* fit in L1, having 256 threads simultaneously requesting data creates pressure. The random/strided access patterns of Global Memory become less efficient than the coordinated access of Shared Memory.
* **The Power of `ldmatrix`:** At this scale, the overhead of setting up Shared Memory is fully amortized. The specialized hardware path of `ldmatrix` (broadcasting data to multiple threads without bank conflicts) provides a throughput that L1 Cache access cannot match under high contention.
* **Manual Shared Memory Fails:** It consistently remains the slowest option. It suffers from the high latency of setting up Shared Memory *plus* the inefficiency of scalar load instructions. It combines the worst of both worlds.

---

## 🧠 Key Takeaways

1.  **Simplicity Wins at Low Scale:** For micro-kernels (1 Warp), relying on the L1 Cache (Global Memory) is superior to building complex pipelines. The overhead of Shared Memory is not justified.
2.  **Alignment Matters:** As data volume grows (4 Warps), the vectorized and aligned nature of `ldmatrix` allows it to match Global Memory performance, despite the extra data movement costs.
3.  **Hardware Acceleration is Necessary for Scaling:** At higher workloads (8 Warps), explicit Shared Memory management with `ldmatrix` is mandatory. It bypasses the contention limits of the L1 Cache and leverages dedicated hardware paths for maximum throughput.

---

## 🛠️ How to Build and Run

**Prerequisites:**
* **Hardware:** NVIDIA Ampere GPU (SM80+).
* **Software:** CUDA Toolkit 11.2+.

**Compilation:**
The project uses C++17 features. Compile using `nvcc`:

```bash
# Compile the scenarios
nvcc -arch=sm_86 -std=c++17 Sparse_TensorCore_Memory_Eval_OneWarp.cu -o benchmark_result_OneWarp
nvcc -arch=sm_86 -std=c++17 Sparse_TensorCore_Memory_Eval_FourWarp.cu -o benchmark_result_FourWarp
nvcc -arch=sm_86 -std=c++17 Sparse_TensorCore_Memory_Eval_EightWarp.cu -o benchmark_result_EightWarp