// src/cuda/forward_kinematics.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return err; \
        } \
    } while(0)

// Constant
#define EPSILON 1e-6f

struct GPUContext {
    // Input buffers (rarely change)
    float *d_length;           // Segment lengths
    float *d_T_cumulative;     // Transformation matrices
    
    // Dynamic buffers (change every frame)
    float *d_kappa;            // Curvatures
    float *d_theta;            // Arc angles
    float *d_phi;              // Bending plane angles
    
    // Output buffer
    float *d_output;           // Spline points
    
    // Configuration
    int num_segments;
    int resolution;
    int total_points;
    
    bool initialized;
};

// Single global context
static GPUContext g_ctx = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, false};

/**
 * CUDA Kernel: Generate spline points for all segments
 * 
 * Each thread computes one point:
 * - Determines which segment and local position
 * - Computes local arc point
 * - Rotates by phi angle
 * - Transforms to global frame
 * - Writes (x, y, z) to output
 */
__global__ void generate_spline_kernel(
    const float* kappa,                 // Curvature values (rad/m)
    const float* theta,                 // Arc angles (degrees)
    const float* phi,                   // Bending plane angles (degrees)
    const float* length,                // Segment lengths (meters)
    const float* T_cumulative,          // Transformation matrices (row-major 4×4)
    int resolution,                     // Number of points to generate per segment
    int num_segments,                   // Number of segments
    float* output                       // Output coordinates
) {
    // Global thread ID = which point to compute
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("[INFO] Thread %d processing point %d\n", threadIdx.x, idx);
    // printf("[INFO] Block %d processing point %d\n", blockIdx.x, idx);
    // printf("[INFO] GridDim.x: %d, BlockDim.x: %d\n", gridDim.x, blockDim.x);

    int total_points = num_segments * resolution;
    // printf("[INFO] Total points to process: %d\n", total_points);

    if (idx >= total_points) return;    // Out of bounds check
    
    int segment_id = idx / resolution;
    // printf("[INFO] Point %d belongs to segment %d\n", idx, segment_id);
    int point_idx = idx % resolution;
    // printf("[INFO] Local point index within segment: %d\n", point_idx);
    
    // Load segment parameters
    float kappa_i = kappa[segment_id];
    float theta_i = theta[segment_id];
    float phi_i = phi[segment_id];
    float length_i = length[segment_id];
    
    // Compute normalized position along segment [0, 1]
    float t = (float)point_idx / (float)(resolution - 1);
    
    // Compute arc length at this position
    float s = t * length_i;
    
    // Compute local arc angle at this position
    float theta_s = t * theta_i;
    
    // ===== STEP 1: Compute local arc point (in segment's local frame) =====
    float x_local, y_local, z_local;
    
    if (fabsf(kappa_i) < EPSILON) {
        // Straight segment (kappa ≈ 0)
        x_local = 0.0f;
        y_local = 0.0f;
        z_local = s;
    } else {
        // Curved segment - circular arc formula
        float radius = 1.0f / kappa_i;
        x_local = radius * (1.0f - cosf(theta_s));
        y_local = 0.0f;  // Arc initially in X-Z plane
        z_local = radius * sinf(theta_s);
    }
    
    // ===== STEP 2: Rotate by phi (bending direction in X-Y plane) =====
    float cos_phi = cosf(phi_i);
    float sin_phi = sinf(phi_i);
    
    float x_rotated = x_local * cos_phi - y_local * sin_phi;
    float y_rotated = x_local * sin_phi + y_local * cos_phi;
    float z_rotated = z_local;
    
    // ===== STEP 3: Transform to global frame using cumulative transformation =====
    // Load transformation matrix for this segment (16 floats, row-major)
    const float* T = &T_cumulative[segment_id * 16];
    
    // Matrix-vector multiplication: P_global = T × [x, y, z, 1]
    // T is 4×4 row-major: [T0 T1 T2 T3] [T4 T5 T6 T7] [T8 T9 T10 T11] [T12 T13 T14 T15]
    float x_global = T[0] * x_rotated + T[1] * y_rotated + T[2] * z_rotated + T[3];
    float y_global = T[4] * x_rotated + T[5] * y_rotated + T[6] * z_rotated + T[7];
    float z_global = T[8] * x_rotated + T[9] * y_rotated + T[10] * z_rotated + T[11];
    
    // ===== STEP 4: Write output (coalesced memory access) =====
    int out_idx = idx * 3;
    output[out_idx + 0] = x_global;
    output[out_idx + 1] = y_global;
    output[out_idx + 2] = z_global;
}

/**
 * Host function: Wrapper for kernel launch
 * Called from Python via ctypes
 */
#ifdef __cplusplus
extern "C" {
#endif

int generate_spline_points(
    const float* h_kappa,           // Host: curvature array [5]
    const float* h_theta,           // Host: arc angle array [5]
    const float* h_phi,             // Host: bending plane angle array [5]
    const float* h_length,          // Host: segment length array [5]
    const float* h_T_cumulative,    // Host: transformation matrices [5 × 16]
    int resolution,                 // Points per segment
    int num_segments,               // Number of segments
    float* h_output                 // Host: output array [total_points × 3]
) {
    // Calculate sizes
    int total_points = num_segments * resolution;
    int output_size = total_points * 3;
    
    // Device pointers
    float *d_kappa, *d_theta, *d_phi, *d_length, *d_T_cumulative, *d_output;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_kappa, num_segments * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_theta, num_segments * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phi, num_segments * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_length, num_segments * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_T_cumulative, num_segments * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_kappa, h_kappa, num_segments * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta, h_theta, num_segments * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi, h_phi, num_segments * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_length, h_length, num_segments * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_T_cumulative, h_T_cumulative, num_segments * 16 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    int threads_per_block = 256;
    int num_blocks = (total_points + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    generate_spline_kernel<<<num_blocks, threads_per_block>>>(
        d_kappa, d_theta, d_phi, d_length, d_T_cumulative,
        resolution, num_segments, d_output
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_kappa);
    cudaFree(d_theta);
    cudaFree(d_phi);
    cudaFree(d_length);
    cudaFree(d_T_cumulative);
    cudaFree(d_output);
    
    return 0;  // Success
}

int initialize_gpu_context(
    int num_segments,
    int resolution,
    const float* h_length,
    const float* h_T_cumulative
) {
    if (g_ctx.initialized) {
        printf("[GPU] Context already initialized, skipping\n");
        return 0;  // Already initialized
    }
    
    g_ctx.num_segments = num_segments;
    g_ctx.resolution = resolution;
    g_ctx.total_points = num_segments * resolution;
    
    // Allocate persistent buffers
    CUDA_CHECK(cudaMalloc(&g_ctx.d_length, num_segments * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_ctx.d_T_cumulative, num_segments * 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_ctx.d_kappa, num_segments * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_ctx.d_theta, num_segments * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_ctx.d_phi, num_segments * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_ctx.d_output, g_ctx.total_points * 3 * sizeof(float)));
    
    // Upload static data
    CUDA_CHECK(cudaMemcpy(g_ctx.d_length, h_length, 
                          num_segments * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_ctx.d_T_cumulative, h_T_cumulative, 
                          num_segments * 16 * sizeof(float), cudaMemcpyHostToDevice));
    
    g_ctx.initialized = true;
    
    printf("[GPU] Context initialized: %d segments × %d points = %d total points\n", 
           num_segments, resolution, g_ctx.total_points);
    
    return 0;
}

/**
 * Fast update function for real-time visualization
 * Call this EVERY TIME slider changes
 */
int update_spline_fast(
    const float* h_kappa,
    const float* h_theta,
    const float* h_phi,
    float* h_output
) {
    if (!g_ctx.initialized) {
        fprintf(stderr, "[GPU] Error: Context not initialized! Call initialize_gpu_context first.\n");
        return -1;
    }
    
    // Upload ONLY the dynamic data (tiny upload!)
    CUDA_CHECK(cudaMemcpy(g_ctx.d_kappa, h_kappa, 
                          g_ctx.num_segments * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_ctx.d_theta, h_theta, 
                          g_ctx.num_segments * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_ctx.d_phi, h_phi, 
                          g_ctx.num_segments * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (g_ctx.total_points + threads_per_block - 1) / threads_per_block;
    
    generate_spline_kernel<<<num_blocks, threads_per_block>>>(
        g_ctx.d_kappa, 
        g_ctx.d_theta, 
        g_ctx.d_phi, 
        g_ctx.d_length,          // Already on GPU!
        g_ctx.d_T_cumulative,    // Already on GPU!
        g_ctx.resolution, 
        g_ctx.num_segments, 
        g_ctx.d_output
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize and copy results back
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, g_ctx.d_output, 
                          g_ctx.total_points * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    
    return 0;
}

/**
 * Update transformation matrices (when geometry changes)
 */
int update_transforms(const float* h_T_cumulative) {
    if (!g_ctx.initialized) {
        fprintf(stderr, "[GPU] Error: Context not initialized!\n");
        return -1;
    }
    
    CUDA_CHECK(cudaMemcpy(g_ctx.d_T_cumulative, h_T_cumulative, 
                          g_ctx.num_segments * 16 * sizeof(float), cudaMemcpyHostToDevice));
    
    return 0;
}

/**
 * Clean up GPU memory
 * Call this when closing application or disabling live preview
 */
int destroy_gpu_context() {
    if (!g_ctx.initialized) {
        return 0;  // Nothing to destroy
    }
    
    cudaFree(g_ctx.d_length);
    cudaFree(g_ctx.d_T_cumulative);
    cudaFree(g_ctx.d_kappa);
    cudaFree(g_ctx.d_theta);
    cudaFree(g_ctx.d_phi);
    cudaFree(g_ctx.d_output);
    
    g_ctx.initialized = false;
    
    printf("[GPU] Context destroyed\n");
    
    return 0;
}

#ifdef __cplusplus
}
#endif