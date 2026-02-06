#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

// 添加错误检查宏（放在文件顶部，include之后）
#define CHECK_CUDA_ERROR(call) {\
    cudaError_t err = (call);\
    if (err != cudaSuccess) {\
        fprintf(stderr, "[CUDA Error] %s:%d | %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
}


/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

template <typename T>
__global__ void trace_kernel(const T* input, T* result, int min_dim, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < min_dim) {
        // 行优先存储中，对角线元素的索引为 i * cols + i
        atomicAdd(result, input[idx * cols + idx]);
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    fprintf(stderr, "[DEBUG] My NEW trace function is called! rows=%zu, cols=%zu\n", rows, cols);
    // 0. 计算基本参数
    size_t num_elements = h_input.size();
    int min_dim = static_cast<int>((rows < cols) ? rows : cols); // 强制转换为int，匹配内核参数
    T h_result = T(0);

    // 1. 分配设备内存
    T *d_input = nullptr, *d_result = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, num_elements * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(T)));

    // 2. 初始化设备结果内存为0，并拷贝输入数据
    CHECK_CUDA_ERROR(cudaMemset(d_result, 0, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(T), cudaMemcpyHostToDevice));

    // 3. 启动内核
    int block_size = 256;
    int grid_size = (min_dim + block_size - 1) / block_size;
    trace_kernel<T><<<grid_size, block_size>>>(d_input, d_result, min_dim, static_cast<int>(cols));
    
    // 4. 检查内核启动和运行是否有错误
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // 关键：等待内核完成

    // 5. 拷贝结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

    // 6. 释放设备内存
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_result));

    return h_result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

template <typename T>
__global__ void flash_attention_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int tgt_len, int src_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal,
    int BLOCK_M, int BLOCK_N) { // 分块大小参数

    // 为当前block分配共享内存，用于存储Q、K、V的块
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* q_tile = reinterpret_cast<T*>(shared_mem);
    T* k_tile = reinterpret_cast<T*>(&q_tile[BLOCK_M * head_dim]);
    T* v_tile = reinterpret_cast<T*>(&k_tile[BLOCK_N * head_dim]);

    int batch = blockIdx.z;
    int query_head = blockIdx.y;
    int kv_head = query_head * kv_heads / query_heads; // GQA: 映射query head到对应的kv head组

    // 1. 初始化当前block负责的输出块O_tile和用于在线softmax的统计量(l_i, m_i)
    T* o_tile = ...; // 在寄存器或共享内存中
    T l_i = 0.0;
    T m_i = -INFINITY;

    // 2. 外层循环: 遍历目标序列的块 (每个block处理一个Q块)
    for (int m_start = 0; m_start < tgt_len; m_start += BLOCK_M) {
        // 3. 从全局内存加载当前Q块到共享内存q_tile
        // 4. 内层循环: 遍历源序列的块 (K, V)
        for (int n_start = 0; n_start < src_len; n_start += BLOCK_N) {
            // 4.1 从全局内存加载当前K块和V块到共享内存k_tile, v_tile
            __syncthreads();

            // 4.2 计算当前块的注意力分数: S_tile = q_tile @ k_tile^T
            // 4.3 如果 is_causal，应用因果掩码 (对于 n_start + j > m_start + i 的位置，S_tile = -inf)
            // 4.4 **在线Softmax更新**:
            //     m_ij = max(row_max_of_S_tile, m_i_old)
            //     P_tile = exp(S_tile - m_ij) （数值稳定的safe softmax[citation:2]）
            //     l_ij = exp(m_i_old - m_ij) * l_i_old + row_sum_of_P_tile
            //     更新输出块: o_tile = diag(exp(m_i_old - m_ij)) * o_tile_old + P_tile @ v_tile
            //     更新统计量: m_i = m_ij, l_i = l_ij
            __syncthreads();
        }
        // 5. 将最终计算好的输出块o_tile写回全局内存O
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int tgt_len, int src_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {

    // 参数校验，分配设备内存，数据拷贝...
    size_t q_size = batch_size * tgt_len * query_heads * head_dim;
    // ... (分配 d_q, d_k, d_v, d_o)

    // 设置分块大小 (需要根据共享内存大小和head_dim调整)
    const int BLOCK_M = 64; // Q块大小
    const int BLOCK_N = 64; // K,V块大小

    // 计算动态共享内存大小
    size_t shared_mem_size = (BLOCK_M * head_dim + 2 * BLOCK_N * head_dim) * sizeof(T);

    // 设置内核执行维度
    dim3 grid(batch_size, query_heads); // 一个block处理一个batch的一个query head
    // 注意：实际需要更精细的网格划分来处理序列长度分块，此处为简化表示

    // 启动内核
    flash_attention_kernel<T><<<grid, num_threads, shared_mem_size>>>(
        d_q, d_k, d_v, d_o,
        batch_size, tgt_len, src_len,
        query_heads, kv_heads, head_dim, is_causal,
        BLOCK_M, BLOCK_N);

    // 设备同步，拷贝结果回主机，释放内存...
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
