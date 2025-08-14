/**
  * Copyright (c) 2021-2024, Systems Group, ETH Zurich
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without modification,
  * are permitted provided that the following conditions are met:
  *
  * 1. Redistributions of source code must retain the above copyright notice,
  * this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  * this list of conditions and the following disclaimer in the documentation
  * and/or other materials provided with the distribution.
  * 3. Neither the name of the copyright holder nor the names of its contributors
  * may be used to endorse or promote products derived from this software
  * without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
  * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
  * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  */

#include <torch/extension.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

// copy memory from raw GPU ptr into a PyTorch tensor
torch::Tensor copy_coyote_ptr(uint64_t ptr, int64_t numel, int device_id) {
    hipSetDevice(device_id);

    void* raw_ptr = reinterpret_cast<void*>(ptr);

    // debug info
    hipPointerAttribute_t attr;
    hipError_t err = hipPointerGetAttributes(&attr, raw_ptr);
    if (err != hipSuccess) {
        std::cerr << "[C++ Wrapper] hipPointerGetAttributes failed: "
                  << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipPointerGetAttributes failed");
    }
    
    // create a dst tensor on GPU with dtype uint32
    auto options = torch::TensorOptions()
                       .dtype(torch::kUInt32)
                       .device(torch::kCUDA, device_id);
    auto dest = torch::empty({numel}, options);

    // copy memory from raw_ptr to dest
    err = hipMemcpy(dest.data_ptr(), raw_ptr, numel * sizeof(uint32_t), hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
        std::cerr << "[C++ Wrapper] hipMemcpy failed: "
                  << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hipMemcpy failed");
    }

    return dest;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_coyote_ptr", &copy_coyote_ptr, "Copy Coyote GPU pointer into PyTorch tensor");
}

