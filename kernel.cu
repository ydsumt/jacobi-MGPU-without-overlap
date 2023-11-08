/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <omp.h>
#include <stdio.h>
//#include "device_launch_parameters.h"
#include <driver_types.h>
//#include "cuda_runtime.h"

#include <windows.h>  
#include <iostream>


#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }

constexpr int MAX_NUM_DEVICES = 32;


typedef float real;
constexpr real tol = 1.0e-8;

const real PI = 2.0 * std::asin(1.0);

//__global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
//    const real pi, const int nx, const int ny) {
//    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < ny; iy += blockDim.x * gridDim.x) {
//        const real y0 = sin(2.0 * pi * iy / (ny - 1));
//        a[iy * nx + 0] = y0;
//        a[iy * nx + (nx - 1)] = y0;
//        a_new[iy * nx + 0] = y0;
//        a_new[iy * nx + (nx - 1)] = y0;
//    }
//}


// I do not understand why there is for loop here.
__global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
    const real pi, const int offset, const int nx,
    const int my_ny, const int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
    real* __restrict__ const l2_norm, const int iy_start,
    const int iy_end, const int nx, const bool calculate_norm, int iy_start_global) {
#ifdef HAVE_CUB
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
#endif  // HAVE_CUB
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    real local_l2_norm = 0.0;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
            a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;

        if (calculate_norm) {
            real residue = new_val - a[iy * nx + ix];
            local_l2_norm += residue * residue;
        }

        //// Debug
        //printf("i = %d, j = %d, a_new = %f, a = %f\n", ix, iy + iy_start_global - iy_start, a_new[iy * nx + ix], a[iy * nx + ix]);
    }
    if (calculate_norm) {
#ifdef HAVE_CUB
        real block_l2_norm = BlockReduce(temp_storage).Sum(local_l2_norm);
        if (0 == threadIdx.y && 0 == threadIdx.x) atomicAdd(l2_norm, block_l2_norm);
#else
        atomicAdd(l2_norm, local_l2_norm);
#endif  // HAVE_CUB
    }
}

double noopt(const int nx, const int ny, const int iter_max, real* const a_ref_h, const int nccheck,
    const bool print);

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 8192);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 8192);
    const bool csv = get_arg(argv, argv + argc, "-csv");
     bool nop2p = get_arg(argv, argv + argc, "-nop2p");

    nop2p = false;
    printf("nop2p: %d \n", nop2p);

    real* a[MAX_NUM_DEVICES];
    real* a_new[MAX_NUM_DEVICES];
    real* a_ref_h;
    real* a_h;
    double runtime_serial = 0.0;

    cudaStream_t compute_stream[MAX_NUM_DEVICES];
    cudaStream_t push_top_stream[MAX_NUM_DEVICES];
    cudaStream_t push_bottom_stream[MAX_NUM_DEVICES];
    cudaEvent_t compute_done[MAX_NUM_DEVICES];
    cudaEvent_t push_top_done[2][MAX_NUM_DEVICES];
    cudaEvent_t push_bottom_done[2][MAX_NUM_DEVICES];

    real* l2_norm_d[MAX_NUM_DEVICES];
    real* l2_norm_h[MAX_NUM_DEVICES];

    int iy_start[MAX_NUM_DEVICES];
    int iy_end[MAX_NUM_DEVICES];

    int iy_start_global[MAX_NUM_DEVICES];

    int chunk_size[MAX_NUM_DEVICES];

    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    // Init memory
    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(0));

        // ny - 2 rows are distributed amongst `size` ranks in such a way
        // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
        // This optimizes load balancing when (ny - 2) % size != 0
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;
        // To calculate the number of ranks that need to compute an extra row,
        // the following formula is derived from this equation:
        // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
        int num_ranks_low = num_devices * chunk_size_low + num_devices -
            (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
        if (dev_id < num_ranks_low)
            chunk_size[dev_id] = chunk_size_low;
        else
            chunk_size[dev_id] = chunk_size_high;

        CUDA_RT_CALL(cudaMalloc(a + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size[dev_id] + 2) * sizeof(real)));

        CUDA_RT_CALL(cudaMemset(a[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size[dev_id] + 2) * sizeof(real)));

        // Calculate local domain boundaries
        int iy_start_global_;  // My start index in the global array
        if (dev_id < num_ranks_low) {
            iy_start_global_ = dev_id * chunk_size_low + 1;
        }
        else {
            iy_start_global_ =
                num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
        }
        iy_start_global[dev_id] = iy_start_global_;


        iy_start[dev_id] = 1;
        iy_end[dev_id] = iy_start[dev_id] + chunk_size[dev_id];

        // Set diriclet boundary conditions on left and right boarder
        initialize_boundaries <<<(ny / num_devices) / 128 + 1, 128 >>> (a[dev_id], 
            a_new[dev_id], PI, iy_start_global_ - 1, nx, (chunk_size[dev_id] + 2), ny);

        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        CUDA_RT_CALL(cudaStreamCreate(compute_stream + dev_id));
        CUDA_RT_CALL(cudaStreamCreate(push_top_stream + dev_id));
        CUDA_RT_CALL(cudaStreamCreate(push_bottom_stream + dev_id));
        CUDA_RT_CALL(cudaEventCreateWithFlags(compute_done + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(push_top_done[0] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(
            cudaEventCreateWithFlags(push_bottom_done[0] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(push_top_done[1] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(
            cudaEventCreateWithFlags(push_bottom_done[1] + dev_id, cudaEventDisableTiming));

        CUDA_RT_CALL(cudaMalloc(l2_norm_d + dev_id, sizeof(real)));
        CUDA_RT_CALL(cudaMallocHost(l2_norm_h + dev_id, sizeof(real)));


        if (!nop2p) {
            const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
            int canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
            printf("Peer Access: %d\n",canAccessPeer);
            if (canAccessPeer) {

                printf("Peer Access is supported\n");
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            }
            const int bottom = (dev_id + 1) % num_devices;
            if (top != bottom) {
                canAccessPeer = 0;
                CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
                if (canAccessPeer) {
                    CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
                }
            }
        }
        CUDA_RT_CALL(cudaDeviceSynchronize());

        //CUDA_RT_CALL(cudaDeviceSynchronize());

    }

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    int iter = 0;
    bool calculate_norm = true;
    real l2_norm = 1.0;

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    ////debug 
    //std::ofstream outfile;
    //outfile.open("C:/Users/yunde.su/work/VIP/MGPU/results/Jacobi_mGPU.txt");

    double start = omp_get_wtime();
    while (l2_norm > tol && iter < iter_max) {
        for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
            const int bottom = (dev_id + 1) % num_devices;
            CUDA_RT_CALL(cudaSetDevice(dev_id));

            CUDA_RT_CALL(
                cudaMemsetAsync(l2_norm_d[dev_id], 0, sizeof(real), compute_stream[dev_id]));

            CUDA_RT_CALL(
                cudaStreamWaitEvent(compute_stream[dev_id], push_top_done[(iter % 2)][bottom], 0));
            CUDA_RT_CALL(
                cudaStreamWaitEvent(compute_stream[dev_id], push_bottom_done[(iter % 2)][top], 0));

            calculate_norm = (iter % nccheck) == 0 || (!csv && (iter % 100) == 0);
            dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
                (chunk_size[dev_id] + dim_block_y - 1) / dim_block_y, 1);


            jacobi_kernel<dim_block_x, dim_block_y>
                << < dim_grid, { dim_block_x, dim_block_y, 1 }, 0, compute_stream[dev_id] >> > (
                    a_new[dev_id], a[dev_id], l2_norm_d[dev_id], iy_start[dev_id], iy_end[dev_id],
                    nx, calculate_norm, iy_start_global[dev_id]);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaEventRecord(compute_done[dev_id], compute_stream[dev_id]));


            // The same stream, memcpy after kernel launch.
            if (calculate_norm) {
                CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h[dev_id], l2_norm_d[dev_id], sizeof(real),
                    cudaMemcpyDeviceToHost, compute_stream[dev_id]));
            }

            // Update ghost cells.
            CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream[dev_id], compute_done[dev_id], 0));
            CUDA_RT_CALL(cudaMemcpyAsync(a_new[top] + (iy_end[top] * nx),
                a_new[dev_id] + iy_start[dev_id] * nx, nx * sizeof(real),
                cudaMemcpyDeviceToDevice, push_top_stream[dev_id]));
            CUDA_RT_CALL(
                cudaEventRecord(push_top_done[((iter + 1) % 2)][dev_id], push_top_stream[dev_id]));

            CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream[dev_id], compute_done[dev_id], 0));
            CUDA_RT_CALL(cudaMemcpyAsync(a_new[bottom], a_new[dev_id] + (iy_end[dev_id] - 1) * nx,
                nx * sizeof(real), cudaMemcpyDeviceToDevice,
                push_bottom_stream[dev_id]));
            CUDA_RT_CALL(cudaEventRecord(push_bottom_done[((iter + 1) % 2)][dev_id],
                push_bottom_stream[dev_id]));
        }
        if (calculate_norm) {
            l2_norm = 0.0;
            for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
                CUDA_RT_CALL(cudaStreamSynchronize(compute_stream[dev_id]));
                l2_norm += *(l2_norm_h[dev_id]);

                //// debug
                //printf("id = % d, norm_l = % f\n", dev_id, *(l2_norm_h[dev_id]));
            }

            l2_norm = std::sqrt(l2_norm);
            if (!csv && (iter % 10) == 0) printf("%5d, %0.6f\n", iter, l2_norm);

            //// debug
            //outfile << iter << " " << l2_norm << std::endl;
        }

        for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
            std::swap(a_new[dev_id], a[dev_id]);
        }
        iter++;
    }

    //// debug
    //outfile.close();

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }
    
    double stop = omp_get_wtime();

    std::cout << "Wall time:  " << stop - start << " seconds.";

    // Free memory
    for (int dev_id = (num_devices - 1); dev_id >= 0; --dev_id) {
        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaEventDestroy(push_bottom_done[1][dev_id]));
        CUDA_RT_CALL(cudaEventDestroy(push_top_done[1][dev_id]));
        CUDA_RT_CALL(cudaEventDestroy(push_bottom_done[0][dev_id]));
        CUDA_RT_CALL(cudaEventDestroy(push_top_done[0][dev_id]));
        CUDA_RT_CALL(cudaEventDestroy(compute_done[dev_id]));
        CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream[dev_id]));
        CUDA_RT_CALL(cudaStreamDestroy(push_top_stream[dev_id]));
        CUDA_RT_CALL(cudaStreamDestroy(compute_stream[dev_id]));

        CUDA_RT_CALL(cudaFreeHost(l2_norm_h[dev_id]));
        CUDA_RT_CALL(cudaFree(l2_norm_d[dev_id]));

        CUDA_RT_CALL(cudaFree(a_new[dev_id]));
        CUDA_RT_CALL(cudaFree(a[dev_id]));
        /*if (0 == dev_id) {
            CUDA_RT_CALL(cudaFreeHost(a_h));
            CUDA_RT_CALL(cudaFreeHost(a_ref_h));
        }*/
    }

    return 1;
    //return result_correct ? 0 : 1;

}
