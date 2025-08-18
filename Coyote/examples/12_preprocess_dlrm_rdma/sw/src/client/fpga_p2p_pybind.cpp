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

#include <any>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <iomanip>
#include <ctime>
#include <utility>

// AMD GPU management & run-time libraries
#include <hip/hip_runtime.h>

// Coyote-specific includes
#include "cBench.hpp"
#include "cThread.hpp"
#include "constants.hpp"

// for pybind11 wrapper
#include <memory>
#include <chrono>
#include <thread>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

constexpr bool const IS_CLIENT = true;

namespace py = pybind11;

class FpgaP2PWrapper {
public:
    int batch_size;
    int num_elements;
    unsigned int size;
    unsigned int n_runs;

    coyote::cThread<std::any> coyote_thread;
    int *mem_cpu;
    int *mem_gpu[2];
    coyote::sgEntry sg;

    // hard-coded the hacc-box-01 IP addr. and operation = WRITE for testing purposes; change if needed
    bool operation = true;
    std::string server_ip = "10.253.74.120"; 

    double total_latency_ns = 0.0;
    double total_bytes = 0.0;

    FpgaP2PWrapper(int batch_size_input) 
	    : coyote_thread(DEFAULT_VFPGA_ID, getpid(), 0)
    {
        batch_size = batch_size_input;
        num_elements = batch_size * 48;
        size = num_elements * sizeof(int);
        n_runs = 1;

        if (hipSetDevice(DEFAULT_GPU_ID)) {
            throw std::runtime_error("Couldn't select GPU!");
        }

        unsigned int max_size = 1 * 1024 * 1024;
	mem_cpu = (int *) coyote_thread.initRDMA(max_size, coyote::defPort, server_ip.c_str());
	mem_gpu[0] = (int *) coyote_thread.getMem({coyote::CoyoteAlloc::GPU, max_size});
	mem_gpu[1] = (int *) coyote_thread.getMem({coyote::CoyoteAlloc::GPU, max_size});
    	sg.rdma = { .len = size };
    }

    // generates synthetic data for a given batch size with similar format to Criteo dataset
    void generate_batch_data() {
        srand(time(NULL));
	if (operation == true) {
    	    for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
        	int base_idx = sample_idx * 48;

        	// --- Packet 1: 1 label + 13 dense + 2 padding ---
        	mem_cpu[base_idx + 0] = rand() % 2; // label 0/1

        	for (int i = 0; i < 13; ++i) {
            	    mem_cpu[base_idx + 1 + i] = rand() % 512 - 256; // dense features
        	}

        	mem_cpu[base_idx + 14] = 0; // padding
        	mem_cpu[base_idx + 15] = 0;

        	// --- Packet 2: 16 sparse ---
        	for (int i = 0; i < 16; ++i) {
            	    mem_cpu[base_idx + 16 + i] = rand() % 100000;
        	}

        	// --- Packet 3: 10 sparse + 6 padding ---
        	for (int i = 0; i < 10; ++i) {
            	    mem_cpu[base_idx + 32 + i] = rand() % 100000;
        	}

        	for (int i = 0; i < 6; ++i) {
            	    mem_cpu[base_idx + 42 + i] = 0; // padding
        	}
    	    }

	} 
    }

    double run_bench(uint transfers, int buffer_idx) {
	auto prep_fn = [&]() {
            coyote_thread.clearCompleted();
            coyote_thread.connSync(IS_CLIENT);
        };
	
	coyote::CoyoteOper coyote_operation = operation ? coyote::CoyoteOper::REMOTE_RDMA_WRITE : coyote::CoyoteOper::REMOTE_RDMA_READ;
    	auto bench_fn = [&]() {
            for (int i = 0; i < transfers; i++) {
                coyote_thread.invoke(coyote_operation, &sg);
            }

            int completed = 0;
            while (completed < transfers) {
            	if (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) > completed) {
                    completed += 1;
                    auto hipMemcpy_result = hipMemcpy(mem_gpu[buffer_idx], mem_cpu, sg.rdma.len, hipMemcpyHostToDevice);
		    if (hipMemcpy_result != hipSuccess) {
                        std::cerr << "DEBUG: hipMemcpy failed!" << std::endl;
                        throw std::runtime_error("hipMemcpy failed!");
                    }
                }
            }
        };

        // execute benchmark
        coyote::cBench bench(n_runs, 0);
        bench.execute(bench_fn, prep_fn);

	return bench.getAvg() / (1. + (double) operation);

    }

    // performs preprocessing for a single batch
    void preprocess_single_batch(int buffer_idx) {
        //sg.local.dst_addr = mem_gpu[buffer_idx];
	
        double latency_time = run_bench(N_THROUGHPUT_REPS, buffer_idx);
        double throughput = ((double)N_THROUGHPUT_REPS * (double)size) / (1024.0 * 1024.0 * latency_time * 1e-9);
        
  	total_latency_ns += latency_time;  // ns
	total_bytes += static_cast<double>(N_THROUGHPUT_REPS) * size;  // bytes
    }

    // returns raw GPU pointer of destination buffer
    uint64_t get_dst_ptr(int buffer_idx) {
        return reinterpret_cast<uint64_t>(mem_gpu[buffer_idx]);
    }

    // returns {total_latency_µs, overall_throughput_MBps}
    std::pair<double, double> get_measurements() const {
        if (total_latency_ns == 0.0)  
            return {0.0, 0.0};

        double tput_MBps = total_bytes / (1024.0 * 1024.0 * total_latency_ns * 1e-9);
        return {total_latency_ns / 1e3, tput_MBps};  // µs, MB/s
    }

};

PYBIND11_MODULE(fpga_p2p_pybind, m) {
    py::class_<FpgaP2PWrapper>(m, "FpgaP2PWrapper")
        .def(py::init<int>())
        .def("get_dst_ptr", &FpgaP2PWrapper::get_dst_ptr, py::arg("buffer_idx"))
        .def("preprocess_single_batch", &FpgaP2PWrapper::preprocess_single_batch, py::arg("buffer_idx"))
        .def("generate_batch_data", &FpgaP2PWrapper::generate_batch_data)
        .def("get_measurements", &FpgaP2PWrapper::get_measurements, "Returns (total_latency_us, overall_throughput_MBps)");
}

