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

#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sample_prob.h"
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/sample_logits_op.h"

namespace paddle {
namespace operators {

DEFINE_bool(debug_print, true, "run debug mode");

// UNDERSTAND: something like take_along_axis in numpy.
template <typename T>
__global__ void GPUTakeAlongD1(size_t size,
                           const int batch_size,
                           const int array_slice_size,
                           const int idx_slice_size,
                           const T* p_array,
                           const int64_t* p_index,
                           T* p_value) {
  const auto value_slice_size = idx_slice_size;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = blockDim.x * gridDim.x;

  for (; idx < size; idx += step_size) {
    int i = idx / idx_slice_size;
    auto array_index = p_index[idx];
    p_value[idx] =
        p_array[i * array_slice_size + array_index];
  }
}

// UNDERSTAND: something like put_along_axis in numpy but if there is duplicate
// indices, scatter is done in += way.
template <typename T>
__global__ void GPUPutAlongD1(size_t size,
                           const int batch_size,
                           const int array_slice_size,
                           const int idx_slice_size,
                           T* p_array,
                           const int64_t* p_index,
                           const T* p_value) {
  const auto value_slice_size = idx_slice_size;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = blockDim.x * gridDim.x;
  
  // size == batch_size
  for (; idx < size; idx += step_size) {
    int i = idx;
    for (int j = 0; j < idx_slice_size; ++j) {
      auto array_index = p_index[i*idx_slice_size + j];
      p_array[i * array_slice_size + array_index] += 
        p_value[i*idx_slice_size + j];
    }
  }
}

// UNDERSTAND: set label as 0,1,...,num_true-1 
template <typename T>
__global__ void GPUSetLabel(size_t size,
                           const int num_true,
                           int64_t* p_array) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = blockDim.x * gridDim.x;

  for (; idx < size; idx += step_size) {
    p_array[idx] = idx % num_true;
  }
}


// UNDERSTAND: compute accidentdal hits from samples and minus corresponding
// logits by a float max, here 1e20
template <typename T>
__global__ void gpu_compute_remove_accidental_hits(const int size,
                           const int num_true,
                           const int idx_slice_size,
                           const int64_t* p_index,
                           T* p_value) {
  const auto value_slice_size = idx_slice_size;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = blockDim.x * gridDim.x;

  for (; idx < size; idx += step_size) {
    int i = idx / idx_slice_size;
    if (idx % idx_slice_size < num_true) continue;
    for (int j = 0; j < num_true; ++j) {
      const auto true_idx = i * idx_slice_size + j;
      if (p_index[true_idx] == p_index[idx]) {
        p_value[idx] -= 1e20;
        break;
      }
    }
  }
}


template <typename T>
class SampleLogitsCUDAKernel : 
      public framework::OpKernel<T> {
 public:
  using Tensor = framework::Tensor;
  template <typename type>
  void Print(Tensor & t, std::string name) const {
    if (FLAGS_debug_print) {
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
  }

  void Compute(const framework::ExecutionContext& context) const override {
    // get necessary inputs
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* label = context.Input<Tensor>("Label");
    VLOG(3) << "Enter SampleLogitsCUDAKernel";

    // get necessary outputs
    Tensor* samples = context.Output<Tensor>("Samples");
    Tensor* sampled_logits = context.Output<Tensor>("SampledLogits");
    Tensor* sampled_label = context.Output<Tensor>("SampledLabel");

    // shapes
    const auto batch_size = logits->dims()[0];
    const auto num_classes = logits->dims()[1];
    const auto label_dim = label->dims();
    const auto num_true = label_dim[1];
    const auto samples_dim = samples->dims();

    // attrs
    const auto num_samples = context.Attr<int>("num_samples");
    const bool use_custom_samples = context.Attr<bool>("use_custom_samples");
    const bool uniq = context.Attr<bool>("uniq");
    const bool remove_accidental_hits =
        context.Attr<bool>("remove_accidental_hits");

    // device contexts
    auto& dev_ctx = context.cuda_device_context();

    // UNDERSTAND: allocate memories for temporaries
    Tensor probabilities_tmp;
    Tensor* probabilities = &probabilities_tmp;
    sampled_logits->mutable_data<T>(samples_dim, context.GetPlace());
    auto sampled_label_data =
        sampled_label->mutable_data<int64_t>(label_dim, context.GetPlace());
    int threads = 512;
    size_t size = batch_size * num_true; 
    int grid = (size + threads - 1) / threads;
    GPUSetLabel<T><<<grid, threads, 0, context.cuda_device_context().stream()>>>(
        size, num_true, sampled_label_data);

    if (use_custom_samples) {
      const Tensor* custom_samples = context.Input<Tensor>("CustomSamples");
      const Tensor* custom_probabilities =
          context.Input<Tensor>("CustomProbabilities");
      samples->ShareDataWith(*custom_samples);
      probabilities->ShareDataWith(*custom_probabilities);
    } else {
      samples->mutable_data<int64_t>(context.GetPlace());
      probabilities->mutable_data<T>(samples_dim, context.GetPlace());
      // UNDERSTAND: sampling
      const auto seed = context.Attr<int>("seed");
      auto sampler_with_prob =
          math::GPUSampleWithProb<T>();
      Print<int64_t>(*samples, std::string("samples1"));
      sampler_with_prob(context.cuda_device_context(), seed, num_classes, uniq,
          num_samples, label, samples, probabilities);
    }
    Print<int64_t>(*samples, std::string("samples2"));
    Print<T>(*probabilities, std::string("probabilities"));

    // UNDERSTAND: gather sampled logits and remove accidental hits if needed
    const auto num_take = samples->dims()[1];
    const auto array_dims = logits->dims();
    const auto idx_dims = samples->dims();

    const T* p_array = logits->data<T>();
    const int64_t* p_index = samples->data<int64_t>();
    T* p_value = sampled_logits->data<T>();
  
    // src slice size
    const auto array_slice_size = array_dims[1];
    // index slice size
    const auto idx_slice_size = idx_dims[1];

    size = batch_size * num_take; 
    grid = (size + threads - 1) / threads;
    GPUTakeAlongD1<T><<<grid, threads, 0, context.cuda_device_context().stream()>>>(
        size, batch_size, array_slice_size, idx_slice_size, p_array, p_index, p_value);
    Print<T>(*sampled_logits, std::string("sampled_logits"));

    if (remove_accidental_hits) {
      const size_t size = batch_size * (num_true + num_samples); 
      int grid = (size + threads - 1) / threads;
      gpu_compute_remove_accidental_hits<T><<<grid, threads, 0, context.cuda_device_context().stream()>>>(size, num_true, idx_slice_size, p_index, p_value);
      Print<T>(*sampled_logits, std::string("sampled_logits_remove_accidental_hits"));
    }

    // subtracted sampled logits with logQ(y|x)
    auto probs = EigenMatrix<T>::From(*probabilities);
    auto smp_logits = EigenMatrix<T>::From(*sampled_logits);
    smp_logits.device(*dev_ctx.eigen_device()) =
        (smp_logits - probs.log().unaryExpr(TolerableValue<T>()))
            .unaryExpr(TolerableValue<T>());
    Print<T>(*sampled_logits, std::string("sampled_logits_res"));

  }
};

template <typename T>
class SampleLogitsGradCUDAKernel : public framework::OpKernel<T> {
 public:
  using Tensor = framework::Tensor;
  template <typename type>
  void Print(const Tensor & t, std::string name) const {
    if (FLAGS_debug_print) {
      return;
    }
    VLOG(1) << "qxz print "<< name;
    VLOG(1) << name << "size = " << t.numel();
    size_t size = t.numel();
    const type *d = t.data<type>();
  #ifdef PADDLE_WITH_CUDA
      std::vector<type> vec;
      platform::DeviceContextPool::Instance().Get(t.place())->Wait();
      if (platform::is_gpu_place(t.place())) {
	vec.resize(size);
	cudaMemcpy(vec.data(), d, sizeof(T) * size, cudaMemcpyDeviceToHost);
	d = vec.data();
      }
  #endif
    VLOG(1) << name << " data_ptr = " << static_cast<const void*>(d);
    std::string out;
    for (size_t i = 0; i < size; i++) {
	 out += std::to_string(d[i]);
	 out += ",";
    }
    VLOG(1) << out;
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto logits_grad = context.Output<Tensor>(framework::GradVarName("Logits"));
    const Tensor* samples = context.Input<Tensor>("Samples");
    const Tensor*  sampled_logits_grad = context.Input<Tensor>(framework::GradVarName("SampledLogits"));
    logits_grad->mutable_data<T>(context.GetPlace());

    auto& dev_ctx = context.cuda_device_context();
    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    set_zero(dev_ctx, logits_grad, static_cast<T>(0));

    // UNDERSTAND: scatter it back to logit_grad
    const auto batch_size = samples->dims()[0];
    const auto num_put = samples->dims()[1];
    const auto array_dims = logits_grad->dims();
    const auto idx_dims = samples->dims();

    T* p_array = logits_grad->data<T>();
    const int64_t* p_index = samples->data<int64_t>();
    const T* p_value = sampled_logits_grad->data<T>();
  
    // src slice size
    const auto array_slice_size = array_dims[1];
    // index slice size
    const auto idx_slice_size = idx_dims[1];

    int threads = 128;
    const size_t size = batch_size; 
    int grid = (size + threads - 1) / threads;

    Print<T>(*sampled_logits_grad, std::string("sampled_logits_grad"));
    Print<int64_t>(*samples, std::string("samples"));
    GPUPutAlongD1<T><<<grid, threads, 0, context.cuda_device_context().stream()>>>(size, batch_size, array_slice_size, idx_slice_size, p_array, p_index, p_value);
    Print<T>(*logits_grad, std::string("logits_grad"));
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(sample_logits,
                       ops::SampleLogitsCUDAKernel<float>,
                       ops::SampleLogitsCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(sample_logits_grad,
                       ops::SampleLogitsGradCUDAKernel<float>,
                       ops::SampleLogitsGradCUDAKernel<double>);
