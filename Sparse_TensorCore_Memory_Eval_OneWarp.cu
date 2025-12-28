#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cuda_bf16.h>

#include <iostream>
#include <random>

#include <cuda_runtime.h>

#include <random>
#include <chrono>
#include <iomanip>

using bfloat16_t = __nv_bfloat16;

__host__ __device__ uint32_t packbf16_2(bfloat16_t fst, bfloat16_t snd)
{
    uint16_t castedf = *reinterpret_cast<uint16_t*>(&fst);
    uint16_t casteds = *reinterpret_cast<uint16_t*>(&snd);

    return ((uint32_t)casteds | (uint32_t)castedf << 16);
}

bool compareAbs(bfloat16_t n1, bfloat16_t n2)
{
    return (std::abs(static_cast<float>(n1)) > std::abs(static_cast<float>(n2)));
}

__host__ void pruning(bfloat16_t *A, int M, int K)
{
    for (int i = 0; i < M*K; i = i + 4)
    {
        int min_val_idx = 0;
        int min_sec_val_idx = 0;

        for (int j = 0; j < 3; j++)
        {
            if (compareAbs(A[i + min_val_idx], A[i + j + 1]))
                min_val_idx = j + 1;
        }

        if (min_val_idx == 0) 
            min_sec_val_idx = 1;

        for (int k = 0; k < 3; k++)
        {
            int candidate_idx = k + 1;

            if (candidate_idx == min_val_idx) continue;

            if (compareAbs(A[i + min_sec_val_idx], A[i + candidate_idx]))
                min_sec_val_idx = candidate_idx;
        }

        A[i + min_val_idx] = bfloat16_t(0);
        A[i + min_sec_val_idx] = bfloat16_t(0);
    }
}

__host__ void compress_and_get_metada(bfloat16_t *A_pruned, int M, int K, bfloat16_t *A_compressed, uint32_t *Metadata_list)
{
    int comp_idx = 0;
    int meta_list_idx = 0;
    
    uint32_t temp_meta_reg = 0;
    int group_count = 0;

    for (int i = 0; i < M * K; i = i + 4)
    {
        int indices[2] = {0, 0};
        int found = 0;

        for (int j = 0; j < 4; j++)
        {
            float val = static_cast<float>(A_pruned[i + j]);

            if (val != 0.0f)
            {
                A_compressed[comp_idx] = A_pruned[i + j];
                comp_idx++;

                if (found < 2)
                {
                    indices[found] = j;
                    found++;
                }
            }
        }

        while (found < 2)
        {
            A_compressed[comp_idx] = bfloat16_t(0);
            comp_idx++;
            indices[found] = 0;
            found++;
        }

        uint32_t pair_meta = (indices[0]) | (indices[1] << 2);

        temp_meta_reg = temp_meta_reg | (pair_meta << (group_count * 4));
        
        group_count++;

        if (group_count == 8)
        {
            Metadata_list[meta_list_idx] = temp_meta_reg;
            meta_list_idx++;
            
            temp_meta_reg = 0;
            group_count = 0;
        }
    }
}

__global__ void simpleOneTensorGemm_GlobalMemAcc(bfloat16_t *A_compressed, bfloat16_t *B, float *C, int32_t *Metadata, long long *d_cycles)
{
    __syncthreads();

    asm volatile ("" ::: "memory"); 

    long long start_clock = 0;
    long long end_clock = 0;

    if (threadIdx.x == 0) {
        start_clock = clock64();
    }

    // th 0,1 ; 4,5 ; 8,9 ; 12,13 ; 16,17 ; 20;21 ; 24,25 ; 28,29
    int idx = threadIdx.x;

    int row_idx = idx / 4; // 0, 1, 2, 3, 4, 5, 6, 7
    int col_idx = idx % 2;  // 0, 1

    uint32_t A_mat_regs[4];
    uint32_t B_mat_regs[4];
    float C_mat_regs[4];

    uint32_t metadata_reg;

    C_mat_regs[0] = 0.0f;
    C_mat_regs[1] = 0.0f;
    C_mat_regs[2] = 0.0f;
    C_mat_regs[3] = 0.0f;

    uint32_t raw_row_low  = Metadata[row_idx];      
    uint32_t raw_row_high = Metadata[row_idx + 8];

    if (col_idx == 0)
    {
        metadata_reg = (raw_row_low & 0xFFFF) | ((raw_row_high & 0xFFFF) << 16);
    }
    else
    {
        metadata_reg = (raw_row_low >> 16) | (raw_row_high & 0xFFFF0000);
    } 

    bfloat16_t *thStartA = A_compressed + (idx % 4) *  2 + (idx / 4) * 16;
    bfloat16_t *thStartB = B            + (idx % 4) * 16 + (idx / 4)     ;
    float            *thStartC = C            + (idx % 4) *  2 + (idx / 4) *  8;

    A_mat_regs[0] = packbf16_2(thStartA[  1], thStartA[  0]);
    A_mat_regs[1] = packbf16_2(thStartA[129], thStartA[128]);
    A_mat_regs[2] = packbf16_2(thStartA[  9], thStartA[  8]);
    A_mat_regs[3] = packbf16_2(thStartA[137], thStartA[136]);

    B_mat_regs[0] = packbf16_2(thStartB[  8], thStartB[  0]);
    B_mat_regs[1] = packbf16_2(thStartB[ 72], thStartB[ 64]);
    B_mat_regs[2] = packbf16_2(thStartB[136], thStartB[128]);
    B_mat_regs[3] = packbf16_2(thStartB[200], thStartB[192]);

    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "      
        "{%4, %5, %6, %7}, "      
        "{%8, %9, %10, %11}, "    
        "{%12, %13, %14, %15}, "  
        "%16, 0x0;\n"             
        : "=f"(C_mat_regs[0]), "=f"(C_mat_regs[1]), "=f"(C_mat_regs[2]), "=f"(C_mat_regs[3])
        : "r"(A_mat_regs[0]), "r"(A_mat_regs[1]), "r"(A_mat_regs[2]), "r"(A_mat_regs[3]),
        "r"(B_mat_regs[0]), "r"(B_mat_regs[1]), "r"(B_mat_regs[2]), "r"(B_mat_regs[3]),
        "f"(C_mat_regs[0]), "f"(C_mat_regs[1]), "f"(C_mat_regs[2]), "f"(C_mat_regs[3]),
        "r"(metadata_reg)
        );

    thStartC[0] = C_mat_regs[0];
    thStartC[1] = C_mat_regs[1];
    thStartC[64] = C_mat_regs[2];
    thStartC[65] = C_mat_regs[3];

    asm volatile ("" ::: "memory");
    __syncthreads();

    if (threadIdx.x == 0)
    {
        end_clock = clock64();
        *d_cycles = *d_cycles + (end_clock - start_clock);
    }
}

__global__ void simpleOneTensorGemm_SharedMemAcc(bfloat16_t *A_compressed, bfloat16_t *B, float *C, int32_t *Metadata, long long *d_cycles)
{
    __syncthreads();

    asm volatile ("" ::: "memory"); 

    long long start_clock = 0;
    long long end_clock = 0;

    if (threadIdx.x == 0) {
        start_clock = clock64();
    }

    // th 0,1 ; 4,5 ; 8,9 ; 12,13 ; 16,17 ; 20;21 ; 24,25 ; 28,29
    int idx = threadIdx.x;

    int row_idx = idx / 4; // 0, 1, 2, 3, 4, 5, 6, 7
    int col_idx = idx % 2;  // 0, 1

    uint32_t A_mat_regs[4];
    uint32_t B_mat_regs[4];
    float C_mat_regs[4];

    uint32_t metadata_reg;

    C_mat_regs[0] = 0.0f;
    C_mat_regs[1] = 0.0f;
    C_mat_regs[2] = 0.0f;
    C_mat_regs[3] = 0.0f;

    __shared__ bfloat16_t smemA[256]; // M x K(compressed) ==> 16 X 16 = 256
    __shared__ bfloat16_t smemB[256]; // K x N             ==> 32 X  8 = 256

    *((uint4*)&smemA[idx * 8]) = *((uint4*)&A_compressed[idx * 8]);    // 128 bit vectorized copy
    *((uint4*)&smemB[idx * 8]) = *((uint4*)&B[idx * 8]);               // 128 bit vectorized copy

    uint32_t raw_row_low  = Metadata[row_idx];
    uint32_t raw_row_high = Metadata[row_idx + 8];

    if (col_idx == 0)
    {
        metadata_reg = (raw_row_low & 0xFFFF) | ((raw_row_high & 0xFFFF) << 16);
    }
    else
    {
        metadata_reg = (raw_row_low >> 16) | (raw_row_high & 0xFFFF0000);
    }

    __syncthreads();   // be sure all threads write the values smem

    bfloat16_t *thStartA = smemA + (idx % 4) *  2 + (idx / 4) * 16;
    bfloat16_t *thStartB = smemB + (idx % 4) * 16 + (idx / 4);
    float            *thStartC =     C + (idx % 4) *  2 + (idx / 4) * 8;

    A_mat_regs[0] = packbf16_2(thStartA[  1], thStartA[  0]);
    A_mat_regs[1] = packbf16_2(thStartA[129], thStartA[128]);
    A_mat_regs[2] = packbf16_2(thStartA[  9], thStartA[  8]);
    A_mat_regs[3] = packbf16_2(thStartA[137], thStartA[136]);

    B_mat_regs[0] = packbf16_2(thStartB[  8], thStartB[  0]);
    B_mat_regs[1] = packbf16_2(thStartB[ 72], thStartB[ 64]);
    B_mat_regs[2] = packbf16_2(thStartB[136], thStartB[128]);
    B_mat_regs[3] = packbf16_2(thStartB[200], thStartB[192]);

    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "      
        "{%4, %5, %6, %7}, "      
        "{%8, %9, %10, %11}, "    
        "{%12, %13, %14, %15}, "  
        "%16, 0x0;\n"             
        : "=f"(C_mat_regs[0]), "=f"(C_mat_regs[1]), "=f"(C_mat_regs[2]), "=f"(C_mat_regs[3])
        : "r"(A_mat_regs[0]), "r"(A_mat_regs[1]), "r"(A_mat_regs[2]), "r"(A_mat_regs[3]),
        "r"(B_mat_regs[0]), "r"(B_mat_regs[1]), "r"(B_mat_regs[2]), "r"(B_mat_regs[3]),
        "f"(C_mat_regs[0]), "f"(C_mat_regs[1]), "f"(C_mat_regs[2]), "f"(C_mat_regs[3]),
        "r"(metadata_reg)
        );

    thStartC[0] = C_mat_regs[0];
    thStartC[1] = C_mat_regs[1];
    thStartC[64] = C_mat_regs[2];
    thStartC[65] = C_mat_regs[3];

    asm volatile ("" ::: "memory");
    __syncthreads();

    if (threadIdx.x == 0)
    {
        end_clock = clock64();
        *d_cycles = *d_cycles + (end_clock - start_clock);
    }
}

__global__ void simpleOneTensorGemm_Wldmatrix(bfloat16_t *A_compressed, bfloat16_t *B, float *C, int32_t *Metadata, long long *d_cycles)
{
    __syncthreads();

    asm volatile ("" ::: "memory"); 

    long long start_clock = 0;
    long long end_clock = 0;

    if (threadIdx.x == 0) {
        start_clock = clock64();
    }
    // th 0,1 ; 4,5 ; 8,9 ; 12,13 ; 16,17 ; 20;21 ; 24,25 ; 28,29
    int idx = threadIdx.x;

    int row_idx = idx / 4; // 0, 1, 2, 3, 4, 5, 6, 7
    int col_idx = idx % 2;  // 0, 1

    uint32_t A_mat_regs[4]; // 8 eleman (16 eleman with out compressed)
    uint32_t B_mat_regs[4]; // 8 eleman
    float C_mat_regs[4];    // 4 eleman
    uint32_t metadata_reg;

    C_mat_regs[0] = 0.0f; 
    C_mat_regs[1] = 0.0f;
    C_mat_regs[2] = 0.0f; 
    C_mat_regs[3] = 0.0f;

    __shared__ bfloat16_t smemA[256];
    __shared__ bfloat16_t smemB[256]; 

    *((uint4*)&smemA[idx * 8]) = *((uint4*)&A_compressed[idx * 8]);    // 128 bit vectorized copy
    *((uint4*)&smemB[idx * 8]) = *((uint4*)&B[idx * 8]);               // 128 bit vectorized copy

    uint32_t raw_row_low  = Metadata[row_idx];      
    uint32_t raw_row_high = Metadata[row_idx + 8];

    if (col_idx == 0)
    {
        metadata_reg = (raw_row_low & 0xFFFF) | ((raw_row_high & 0xFFFF) << 16);
    }
    else
    {
        metadata_reg = (raw_row_low >> 16) | (raw_row_high & 0xFFFF0000);
    } 

    __syncthreads();   // be sure all threads write the values smem

    uint32_t smemAint = static_cast<uint32_t>(__cvta_generic_to_shared(smemA + (idx / 16) * 8 + (idx % 16) * 16));
    
    uint32_t smemBint = static_cast<uint32_t>(__cvta_generic_to_shared(smemB + idx * 8));

    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(A_mat_regs[0]), "=r"(A_mat_regs[1]), "=r"(A_mat_regs[2]), "=r"(A_mat_regs[3])
        :  "r"(smemAint));

    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.trans.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(B_mat_regs[0]), "=r"(B_mat_regs[1]), "=r"(B_mat_regs[2]), "=r"(B_mat_regs[3])
        :  "r"(smemBint));


    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "      
        "{%4, %5, %6, %7}, "      
        "{%8, %9, %10, %11}, "    
        "{%12, %13, %14, %15}, "  
        "%16, 0x0;\n"             
        : "=f"(C_mat_regs[0]), "=f"(C_mat_regs[1]), "=f"(C_mat_regs[2]), "=f"(C_mat_regs[3])
        : "r"(A_mat_regs[0]), "r"(A_mat_regs[1]), "r"(A_mat_regs[2]), "r"(A_mat_regs[3]),
        "r"(B_mat_regs[0]), "r"(B_mat_regs[1]), "r"(B_mat_regs[2]), "r"(B_mat_regs[3]),
        "f"(C_mat_regs[0]), "f"(C_mat_regs[1]), "f"(C_mat_regs[2]), "f"(C_mat_regs[3]),
        "r"(metadata_reg)
        );

    float *thStartC = C + (idx % 4) * 2 + (idx / 4) * 8;

    thStartC[0] = C_mat_regs[0];
    thStartC[1] = C_mat_regs[1];
    thStartC[64] = C_mat_regs[2];
    thStartC[65] = C_mat_regs[3];

    asm volatile ("" ::: "memory");
    __syncthreads();

    if (threadIdx.x == 0)
    {
        end_clock = clock64();
        *d_cycles = *d_cycles + (end_clock - start_clock);
    }
}

void cpu_gemm(const bfloat16_t* A, const bfloat16_t* B, float* C,
                 int M, int N, int K)
{
    // A row-major (M x K)
    // B row-major (K x N)
    // C row-major (M x N)

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = static_cast<float>(A[m*K + k]);  // A(m,k) row-major
                float b = static_cast<float>(B[k*N + n]);  // B(k,n) row-major
                acc += a * b;
            }
            C[m*N + n] = acc;
        }
    }
}

const double GPU_FREQ_GHZ = 1.50; 

// --- HELPER FUNCTION: RESULT VERIFICATION ---
bool verify_result(float* d_C, const std::vector<float>& h_C_Ref, int M, int N) {
    std::vector<float> h_C_GPU(M * N);
    cudaMemcpy(h_C_GPU.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    int errors = 0;
    float max_diff = 0.0f;
    float epsilon = 1e-2f; // Tolerance for BF16 precision loss

    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(h_C_GPU[i] - h_C_Ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > epsilon) {
            errors++;
            if (errors <= 3) { // Print first 3 errors only
                printf("  [ERROR] Index %d: GPU=%f, CPU=%f, Diff=%f\n", i, h_C_GPU[i], h_C_Ref[i], diff);
            }
        }
    }

    if (errors > 0) {
        printf("  -> STATUS: FAILED! Total Errors: %d, Max Diff: %f\n", errors, max_diff);
        return false;
    } else {
        printf("  -> STATUS: PASSED! (Max Diff: %f)\n", max_diff);
        return true;
    }
}

// --- BENCHMARK FUNCTION (UPDATED FOR CLOCK64) ---
template <typename KernelFunc>
void run_benchmark(const char* kernel_name, KernelFunc kernel, 
                   bfloat16_t* d_A, bfloat16_t* d_B, float* d_C, int32_t* d_Meta, 
                   const std::vector<float>& h_C_Ref, int M, int N, int K) 
{
    printf("\n=======================================================\n");
    printf("TEST: %s\n", kernel_name);
    printf("=======================================================\n");
    
    // Cycle sayacı için bellek ayır
    long long* d_cycles;
    cudaMalloc(&d_cycles, sizeof(long long));
    cudaMemset(d_cycles, 0, sizeof(long long)); // Temizle

    // measure_iters: Launch overhead çok yüksek olduğu için 5 milyon yeterli.
    int warmup_iters = 10000;
    int measure_iters = 5000000;

    // 1. VERIFICATION
    // Kernel<<<1, 32>>> çağırıyoruz (Bu dosya için 1 Warp yeterli)
    cudaMemset(d_C, 0, M * N * sizeof(float)); 
    kernel<<<1, 32>>>(d_A, d_B, d_C, d_Meta, d_cycles);  
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  [CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_cycles);
        return;
    }

    if (!verify_result(d_C, h_C_Ref, M, N)) {
        printf("  [WARNING] Verification failed! Performance metrics might be invalid.\n");
    }

    // 2. WARMUP
    for(int i=0; i<warmup_iters; i++) {
        kernel<<<1, 32>>>(d_A, d_B, d_C, d_Meta, d_cycles);
    }
    cudaDeviceSynchronize();

    // *** KRİTİK ADIM ***
    // Ölçüm öncesi sayacı SIFIRLIYORUZ.
    cudaMemset(d_cycles, 0, sizeof(long long));

    // 3. PERFORMANCE BENCHMARK
    printf("  -> Running %d iterations...\n", measure_iters);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i=0; i<measure_iters; i++) {
        // Kernel d_cycles üzerine toplama (accumulation) yapacak
        kernel<<<1, 32>>>(d_A, d_B, d_C, d_Meta, d_cycles);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // --- HOST SIDE METRICS (Launch Overhead Dahil) ---
    double avg_time_sec = (milliseconds / 1000.0) / measure_iters;
    
    // GFLOPS Formula: (2 * M * N * K)
    double total_flops_per_launch = 2.0 * (double)M * (double)N * (double)K;
    double gflops_host = (total_flops_per_launch * 1e-9) / avg_time_sec;

    // --- DEVICE SIDE METRICS (Saf Donanım Hızı) ---
    long long total_cycles = 0;
    cudaMemcpy(&total_cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost);

    double avg_cycles = (double)total_cycles / measure_iters;
    double hardware_latency_ns = avg_cycles / GPU_FREQ_GHZ;
    
    // GFLOPS (Hardware): Launch overhead olmadan teorik hız
    double gflops_hardware = (total_flops_per_launch * 1e-9) / (hardware_latency_ns * 1e-9);

    printf("\n  [HOST TIMER - Includes Overhead]\n");
    printf("  -> Duration (Avg)  : %.6f ms (%.2f ns)\n", milliseconds / measure_iters, avg_time_sec * 1e9);
    printf("  -> Performance     : %.4f GFLOPS\n", gflops_host);

    printf("\n  [DEVICE CLOCK64 - Execution Only]\n");
    printf("  -> Avg Cycles      : %.2f cycles\n", avg_cycles);
    printf("  -> Hardware Latency: %.2f ns (assuming %.2f GHz)\n", hardware_latency_ns, GPU_FREQ_GHZ);
    printf("  -> Core Perf.      : %.4f GFLOPS (Theoretical Peak)\n", gflops_hardware);

    printf("\n  [OVERHEAD ANALYSIS]\n");
    double overhead_ns = (avg_time_sec * 1e9) - hardware_latency_ns;
    printf("  -> Launch Overhead : %.2f ns per kernel (%.2f%% of total time)\n", 
           overhead_ns, (overhead_ns / (avg_time_sec * 1e9)) * 100.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_cycles);
}

int main() 
{
    // --- 1. PARAMETERS ---
    const int M = 16;
    const int N = 8;
    const int K = 32;

    printf("--- GRADUATION PROJECT: SPARSE TENSOR CORE BENCHMARK ---\n");
    printf("Matrix Sizes: M=%d, N=%d, K=%d\n", M, N, K);

    // Memory sizes
    int size_A_dense = M * K;
    int size_A_compressed = M * K / 2; // 2:4 sparsity
    int size_B = K * N;
    int size_C = M * N;
    int size_Metadata = (M * K) / 32;

    // --- 2. HOST DATA PREPARATION ---
    std::vector<bfloat16_t> h_A_dense(size_A_dense);
    std::vector<bfloat16_t> h_A_compressed(size_A_compressed);
    std::vector<bfloat16_t> h_B(size_B);
    std::vector<float> h_C_CPU(size_C);
    std::vector<uint32_t> h_Metadata(size_Metadata);

    // Random Number Generation
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> distrib(-1.0f, 1.0f);

    printf("Host: Generating random data...\n");
    for(int i=0; i<size_A_dense; ++i) h_A_dense[i] = static_cast<bfloat16_t>(distrib(gen));
    for(int i=0; i<size_B; ++i) h_B[i] = static_cast<bfloat16_t>(distrib(gen));

    // Pruning and Compression (CPU Simulation)
    printf("Host: Performing Pruning and Compression...\n");
    pruning(h_A_dense.data(), M, K);
    compress_and_get_metada(h_A_dense.data(), M, K, h_A_compressed.data(), h_Metadata.data());

    // Golden Output (CPU Reference)
    printf("Host: Running Reference GEMM...\n");
    cpu_gemm(h_A_dense.data(), h_B.data(), h_C_CPU.data(), M, N, K);

    // --- 3. DEVICE PREPARATION ---
    bfloat16_t *d_A_comp, *d_B;
    float *d_C;
    int32_t *d_Metadata;

    cudaMalloc(&d_A_comp, size_A_compressed * sizeof(bfloat16_t));
    cudaMalloc(&d_B, size_B * sizeof(bfloat16_t));
    cudaMalloc(&d_C, size_C * sizeof(float));
    cudaMalloc(&d_Metadata, size_Metadata * sizeof(int32_t));

    cudaMemcpy(d_A_comp, h_A_compressed.data(), size_A_compressed * sizeof(bfloat16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(bfloat16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Metadata, h_Metadata.data(), size_Metadata * sizeof(int32_t), cudaMemcpyHostToDevice);

    // --- 4. RUN BENCHMARK ---
    // Kernel parametreleri run_benchmark içinde d_cycles eklenerek çağrılıyor.

    // Test 1: Only Global Memory
    run_benchmark("1. Global Memory Access Only", 
                  simpleOneTensorGemm_GlobalMemAcc, 
                  d_A_comp, d_B, d_C, d_Metadata, 
                  h_C_CPU, M, N, K);

    // Test 2: Shared Memory (Manual Load)
    run_benchmark("2. Shared Memory (Manual Load)", 
                  simpleOneTensorGemm_SharedMemAcc, 
                  d_A_comp, d_B, d_C, d_Metadata, 
                  h_C_CPU, M, N, K);

    // Test 3: ldmatrix 
    run_benchmark("3. Shared Memory + LDMATRIX", 
                  simpleOneTensorGemm_Wldmatrix, 
                  d_A_comp, d_B, d_C, d_Metadata, 
                  h_C_CPU, M, N, K);


    cudaFree(d_A_comp);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Metadata);

    printf("\nAll tests completed successfully.\n");
    return 0;
}