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

__global__ void simpleOneTensorGemm_GlobalMemAcc(bfloat16_t *A_compressed, bfloat16_t *B, float *C, int32_t *Metadata, int kernel_loops)
{
    // th 0,1 ; 4,5 ; 8,9 ; 12,13 ; 16,17 ; 20;21 ; 24,25 ; 28,29
    int idx = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;

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

    int A_warp_start_idx = warp_idx / 2;
    int B_warp_start_idx = warp_idx % 2;

    for (int i = 0; i < kernel_loops; i++)
    {
        uint32_t raw_row_low  = Metadata[A_warp_start_idx * 16 + row_idx];      
        uint32_t raw_row_high = Metadata[A_warp_start_idx * 16 + row_idx + 8];

        if (col_idx == 0)
        {
            metadata_reg = (raw_row_low & 0xFFFF) | ((raw_row_high & 0xFFFF) << 16);
        }
        else
        {
            metadata_reg = (raw_row_low >> 16) | (raw_row_high & 0xFFFF0000);
        } 

        bfloat16_t *thStartA = A_compressed + A_warp_start_idx * 256 + (idx % 4) *  2 + (idx / 4) * 16;
        bfloat16_t *thStartB = B            + B_warp_start_idx *   8 + (idx % 4) * 32 + (idx / 4)     ;
        float            *thStartC = C            + (idx % 4) *  2 + (idx / 4) *  16 + A_warp_start_idx * 256 + B_warp_start_idx * 8;

        A_mat_regs[0] = packbf16_2(thStartA[  1], thStartA[  0]);
        A_mat_regs[1] = packbf16_2(thStartA[129], thStartA[128]);
        A_mat_regs[2] = packbf16_2(thStartA[  9], thStartA[  8]);
        A_mat_regs[3] = packbf16_2(thStartA[137], thStartA[136]);

        B_mat_regs[0] = packbf16_2(thStartB[ 16], thStartB[  0]);
        B_mat_regs[1] = packbf16_2(thStartB[144], thStartB[128]);
        B_mat_regs[2] = packbf16_2(thStartB[272], thStartB[256]);
        B_mat_regs[3] = packbf16_2(thStartB[400], thStartB[384]);

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

        thStartC[  0] = C_mat_regs[0];
        thStartC[  1] = C_mat_regs[1];
        thStartC[128] = C_mat_regs[2];
        thStartC[129] = C_mat_regs[3];
    }
}

__global__ void simpleOneTensorGemm_SharedMemAcc(bfloat16_t *A_compressed, bfloat16_t *B, float *C, int32_t *Metadata, int kernel_loops)
{
    // th 0,1 ; 4,5 ; 8,9 ; 12,13 ; 16,17 ; 20;21 ; 24,25 ; 28,29
    int idx = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;

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

    int A_warp_start_idx = warp_idx / 2;
    int B_warp_start_idx = warp_idx % 2;

    __shared__ bfloat16_t smemA[512]; // M x K(compressed) ==> 32 X 16 = 512
    __shared__ bfloat16_t smemB[512]; // K x N             ==> 32 X 16 = 512

    for (int i = 0; i < kernel_loops; i++)
    {
        *((uint2*)&smemA[threadIdx.x * 4]) = *((uint2*)&A_compressed[threadIdx.x * 4]);    // 64 bit vectorized copy
        *((uint2*)&smemB[threadIdx.x * 4]) = *((uint2*)&B[threadIdx.x * 4]);               // 64 bit vectorized copy

        uint32_t raw_row_low  = Metadata[A_warp_start_idx * 16 + row_idx];      
        uint32_t raw_row_high = Metadata[A_warp_start_idx * 16 + row_idx + 8];

        if (col_idx == 0)
        {
            metadata_reg = (raw_row_low & 0xFFFF) | ((raw_row_high & 0xFFFF) << 16);
        }
        else
        {
            metadata_reg = (raw_row_low >> 16) | (raw_row_high & 0xFFFF0000);
        }

        __syncthreads();   // be sure all threads write the values smem

        bfloat16_t *thStartA = smemA + A_warp_start_idx * 256 + (idx % 4) *  2 + (idx / 4) * 16;
        bfloat16_t *thStartB = smemB + B_warp_start_idx *   8 + (idx % 4) * 32 + (idx / 4)     ;
        float            *thStartC = C     + (idx % 4) *  2 + (idx / 4) *  16 + A_warp_start_idx * 256 + B_warp_start_idx * 8;

        A_mat_regs[0] = packbf16_2(thStartA[  1], thStartA[  0]);
        A_mat_regs[1] = packbf16_2(thStartA[129], thStartA[128]);
        A_mat_regs[2] = packbf16_2(thStartA[  9], thStartA[  8]);
        A_mat_regs[3] = packbf16_2(thStartA[137], thStartA[136]);

        B_mat_regs[0] = packbf16_2(thStartB[ 16], thStartB[  0]);
        B_mat_regs[1] = packbf16_2(thStartB[144], thStartB[128]);
        B_mat_regs[2] = packbf16_2(thStartB[272], thStartB[256]);
        B_mat_regs[3] = packbf16_2(thStartB[400], thStartB[384]);

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

        thStartC[  0] = C_mat_regs[0];
        thStartC[  1] = C_mat_regs[1];
        thStartC[128] = C_mat_regs[2];
        thStartC[129] = C_mat_regs[3];
    }
}

__global__ void simpleOneTensorGemm_Wldmatrix(bfloat16_t *A_compressed, bfloat16_t *B, float *C, int32_t *Metadata, int kernel_loops)
{
    // th 0,1 ; 4,5 ; 8,9 ; 12,13 ; 16,17 ; 20;21 ; 24,25 ; 28,29
    int idx = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;

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

    int A_warp_start_idx = warp_idx / 2;
    int B_warp_start_idx = warp_idx % 2;

    __shared__ bfloat16_t smemA[512]; // M x K(compressed) ==> 32 X 16 = 512
    __shared__ bfloat16_t smemB[512]; // K x N             ==> 32 X 16 = 512

    for (int i = 0; i < kernel_loops; i++)
    {
        *((uint2*)&smemA[threadIdx.x * 4]) = *((uint2*)&A_compressed[threadIdx.x * 4]);    // 64 bit vectorized copy
        *((uint2*)&smemB[threadIdx.x * 4]) = *((uint2*)&B[threadIdx.x * 4]);               // 64 bit vectorized copy

        uint32_t raw_row_low  = Metadata[A_warp_start_idx * 16 + row_idx];      
        uint32_t raw_row_high = Metadata[A_warp_start_idx * 16 + row_idx + 8];

        if (col_idx == 0)
        {
            metadata_reg = (raw_row_low & 0xFFFF) | ((raw_row_high & 0xFFFF) << 16);
        }
        else
        {
            metadata_reg = (raw_row_low >> 16) | (raw_row_high & 0xFFFF0000);
        } 

        __syncthreads();   // be sure all threads write the values smem

        uint32_t smemAint = static_cast<uint32_t>(__cvta_generic_to_shared(smemA + A_warp_start_idx * 256 + (idx / 16) * 8 + (idx % 16) * 16));
        
        uint32_t smemBint = static_cast<uint32_t>(__cvta_generic_to_shared(smemB + B_warp_start_idx * 8 + idx * 16));

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

        float *thStartC = C + (idx % 4) *  2 + (idx / 4) *  16 + A_warp_start_idx * 256 + B_warp_start_idx * 8;

        thStartC[  0] = C_mat_regs[0];
        thStartC[  1] = C_mat_regs[1];
        thStartC[128] = C_mat_regs[2];
        thStartC[129] = C_mat_regs[3];
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

// --- BENCHMARK FUNCTION ---
template <typename KernelFunc>
void run_benchmark(const char* kernel_name, KernelFunc kernel, 
                   bfloat16_t* d_A, bfloat16_t* d_B, float* d_C, int32_t* d_Meta, 
                   const std::vector<float>& h_C_Ref, int M, int N, int K) 
{
    printf("\n=======================================================\n");
    printf("TEST: %s\n", kernel_name);
    printf("=======================================================\n");

    // SETTINGS
    // kernel_loops: How many times the matrix multiplication runs INSIDE the GPU kernel.
    // This increases Arithmetic Intensity and hides launch latency.
    int kernel_loops = 500000; 
    
    // measure_iters: How many times we launch the kernel from Host.
    // Since the kernel is now heavy, we can reduce this number.
    int warmup_iters = 10;
    int measure_iters = 100;

    // 1. VERIFICATION
    // We run the kernel with loop_count = 1 just to check correctness.
    cudaMemset(d_C, 0, M * N * sizeof(float)); 
    kernel<<<1, 128>>>(d_A, d_B, d_C, d_Meta, 1);  
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  [CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    if (!verify_result(d_C, h_C_Ref, M, N)) {
        printf("  [WARNING] Verification failed! Performance metrics might be invalid.\n");
    }

    // 2. WARMUP
    for(int i=0; i<warmup_iters; i++) {
        kernel<<<1, 128>>>(d_A, d_B, d_C, d_Meta, kernel_loops);
    }
    cudaDeviceSynchronize();

    // 3. PERFORMANCE BENCHMARK
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i=0; i<measure_iters; i++) {
        // Launch kernel with heavy workload
        kernel<<<1, 128>>>(d_A, d_B, d_C, d_Meta, kernel_loops);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Metrics Calculation
    double avg_time_sec = (milliseconds / 1000.0) / measure_iters;
    
    // GFLOPS Calculation:
    // Formula: (2 * M * N * K) * (Operations inside Kernel)
    double total_flops_per_launch = 2.0 * (double)M * (double)N * (double)K * (double)kernel_loops;
    double gflops = (total_flops_per_launch * 1e-9) / avg_time_sec;

    printf("  -> Kernel Loops   : %d\n", kernel_loops);
    printf("  -> Duration (Avg) : %.6f ms\n", milliseconds / measure_iters);
    printf("  -> Performance    : %.4f GFLOPS\n", gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() 
{
    // --- 1. PARAMETERS ---
    const int M = 32;
    const int N = 16;
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