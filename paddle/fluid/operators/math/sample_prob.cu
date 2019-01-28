/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <iostream>
#include <vector>
#include <thrust/random.h>
#include <thrust/sort.h>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sample_prob.h"

namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;

template <typename T>
__device__ T gpu_adjust_prob(const T prob, const int num_samples, const int num_tries) {
  if (num_samples == num_tries) {
    return prob * num_samples;
  } else {
    return -expm1(num_tries * log1p(-prob));
  }
}

class GPULogUniformSampler{
 public:
  __device__ int64_t Sample(float random, const int range, const float log_range) const;
  __device__ float Probability(int64_t value, const float log_range) const;
};

__device__ int64_t GPULogUniformSampler::Sample(float random, const int range, const float log_range) const {
  // Got Log Uniform distribution from uniform distribution by
  // inverse_transform_sampling method
  const int64_t value =
      static_cast<int64_t>(exp(random * log_range)) - 1;
  // Mathematically, value should be <= range_, but might not be due to some
  // floating point roundoff, so we mod by range_.
  return value % range;
}

__device__ float GPULogUniformSampler::Probability(int64_t value, const float log_range) const {
  // Given f(x) = 1/[(x+1) * log_range_]
  // The value's  probability  is integral of f(x) from value to (value + 1)
  return (log((value + 2.0) / (value + 1.0))) / log_range;
}


template<typename T>
__global__ void SamplingCondidate(const size_t n, const int* num_tries, const int num_true, const std::size_t num_samples, 
    const int64_t* label_data, int64_t* samples_data, T* probabilities_data) {
  const int num_sampled_classes = num_true + num_samples;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = 0;

  for (; idx < n; idx += blockDim.x * gridDim.x) {
    int col_idx = idx % num_sampled_classes;
    int row_idx = idx / num_sampled_classes;
    if (col_idx < num_true) {
      samples_data[idx] = label_data[row_idx * num_true + col_idx];
    } else {
      samples_data[idx] = samples_data[col_idx];
    }
    probabilities_data[idx] = probabilities_data[col_idx];
    probabilities_data[idx] = gpu_adjust_prob(
        probabilities_data[idx], num_samples, num_tries[0]);
  } 
}

template<typename T>
__global__ void UniqSampler(const size_t n, const int seed, const int range, 
    const float log_range, const int num_true, const std::size_t num_samples, int* num_tries, bool* dict_data,
    int64_t* samples_data, T* probabilities_data) {
  thrust::minstd_rand rng;
  rng.seed(seed);
  thrust::uniform_real_distribution<float> dist(0, 1);
  const int num_sampled_classes = num_true + num_samples;

  GPULogUniformSampler sampler;

  int row_idx = 0;
   
  // n == 1
  int j = num_true;

  while (j < num_samples + num_true) {
    ++(*num_tries);
    int idx = row_idx * num_sampled_classes + j;
    auto v = sampler.Sample(dist(rng), range, log_range);
    if (dict_data[v] == false) {
      samples_data[idx] = v;
      dict_data[v] = true;
      probabilities_data[idx] = sampler.Probability(samples_data[idx], range);
      ++j;
    }
  }
}
/*
template <typename T>
void Print(Tensor & t, std::string name) {
  if (!FLAGS_debug_print) {
    return;
  }
  VLOG(1) << "qxz print "<< name;
  VLOG(1) << name << "size = " << t.numel();
  size_t size = t.numel();
  type *d = t.data<type>();
#ifdef PADDLE_WITH_CUDA
    std::vector<type> vec;
    platform::DeviceContextPool::Instance().Get(t.place())->Wait();
    if (platform::is_gpu_place(t.place())) {
      vec.resize(size);
      cudaMemcpy(vec.data(), d, sizeof(T) * size, cudaMemcpyDeviceToHost);
      d = vec.data();
    }
#endif
  VLOG(1) << name << " data_ptr = " << static_cast<void*>(d);
  std::string out;
  for (size_t i = 0; i < size; i++) {
       out += std::to_string(d[i]);
       out += ",";
  }
  VLOG(1) << out;
}*/

template <typename T>
void GPUSampleWithProb<T>::operator()(const platform::CUDADeviceContext& context, const int seed, const int dict_size, const bool uniq,
                  const std::size_t num_samples, const Tensor* L, Tensor* S,
                  Tensor* P) {
    // UNDERSTAND: dimension issues
    const auto lbl_dim = L->dims();
    const int batch_size = lbl_dim[0];
    const int num_true = lbl_dim[1];
    const int num_sampled_classes = num_true + num_samples;
    framework::DDim ret_dim{batch_size, num_sampled_classes};

    // UNDERSTAND: raw data view
    const int64_t* label_data = L->data<int64_t>();
    int64_t* samples_data =
        S->mutable_data<int64_t>(ret_dim, context.GetPlace());
    T* probabilities_data = P->mutable_data<T>(ret_dim, context.GetPlace());
    
    framework::DDim dict_dim{dict_size};
    Tensor dict;
    bool* dict_data =
      dict.mutable_data<bool>(dict_dim, context.GetPlace()); 
    math::SetConstant<platform::CUDADeviceContext, bool> set_zero;
    set_zero(context, &dict, false);

    framework::DDim num_tries_dim{1};
    Tensor num_tries;
    int* num_tries_data =
      num_tries.mutable_data<int>(num_tries_dim, context.GetPlace()); 
    math::SetConstant<platform::CUDADeviceContext, int> set_zero2;
    set_zero2(context, &num_tries, static_cast<int>(0));

    int threads = 1;
    int grid = 1;
    size_t size= 1;
    UniqSampler<T><<<grid, threads, 0, context.stream()>>>(size, seed, dict_size, log(dict_size), num_true, num_samples, num_tries_data, dict_data, samples_data, probabilities_data);
      
    //Print<int>(num_tries, "num_tries");
    threads = 512;
    size = batch_size * num_sampled_classes; 
    grid = (batch_size * num_sampled_classes + threads - 1) / threads;
    SamplingCondidate<T><<<grid, threads, 0, context.stream()>>>(size, num_tries_data, num_true, num_samples, label_data, samples_data, probabilities_data);
};

template class GPUSampleWithProb<float>;
template class GPUSampleWithProb<double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
