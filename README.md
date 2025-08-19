#  FPGA-GPU Training System for Recommender Models

This repository contains the implementation and integration of FPGA-based preprocessing with Meta's [Deep Learning Recommendation Model (DLRM)](https://github.com/facebookresearch/dlrm/tree/main) using the [Coyote](https://github.com/fpgasystems/Coyote/tree/software-cleanup) framework. The goal of this project is to accelerate data preprocessing on FPGA and directly transfer preprocessed data to the GPU for efficient DLRM training.

## Requirements

- **Hardware:**  
  - AMD GPU with HIP support (tested on gfx90a architecture)  
  - Xilinx U55C FPGA device  
- **Software:**  
  - AMD-compatible PyTorch
  - HIP development tools (hipcc compiler)  
  - CMake  
  - pybind11 Python bindings  
  - Python 3.10 or higher  


## Repository Structure

- `Coyote/` — Contains the FPGA preprocessing hardware design, drivers, and software interfaces based on the Coyote framework.  
- `DLRM/` — Contains the adapted DLRM codebase configured to use the FPGA-preprocessed data for training.
- `Measurements/` — Contains performance measurements for the baseline CPU–GPU pipeline and the proposed FPGA–GPU pipeline.


## Compilation and Setup Steps

All commands assume starting from the **root directory** of the repository.

### Step 1: Compile the FPGA bitstream

Compile the FPGA hardware preprocessing pipeline bitstream for the U55C device.  
This step synthesizes the FPGA design and generates the bitstream file.

```bash
cd Coyote/examples/11_preprocess_dlrm/hw
mkdir build_hw && cd build_hw
cmake ../ -DFDEV_NAME=u55c
make project && make bitgen
```
The bitstream will be available in the `build_hw/bitstreams/` folder.


### Step 2: Build and insert the FPGA driver

Compile the kernel driver required for FPGA communication and load it into the kernel.

```bash
cd Coyote/driver
make
```


### Step 3: Compile the Pointer-to-Tensor Python extension

This wrapper exposes GPU pointer allocated by Coyote as Pytorch tensors. 
Important: Make sure to unset CXX environment variable before compiling to avoid build errors.

```bash
cd Coyote/examples/11_preprocess_dlrm/pointer_to_tensor
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH
python3 setup.py build_ext --inplace
```
This produces a `.so` shared object file inside the pointer_to_tensor folder.


### Step 4: Compile the FPGA preprocessing software interface

Build the C++/Python interface to run and manage FPGA preprocessing, using HIP for AMD GPU support.

```bash
cd Coyote/examples/11_preprocess_dlrm/sw
mkdir build && cd build
export CXX=hipcc
cmake ../ -DEN_GPU=1 -DAMD_GPU=gfx90a -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make
```
This creates a shared library `.so` file in the build folder.


### Step 5: Program the FPGA with the bitstream and driver

Load the generated bitstream onto the FPGA and insert the compiled driver module.

```bash
cd Coyote
bash util/program_hacc_local.sh examples/11_preprocess_dlrm/hw/build_hw/bitstreams/cyt_top.bit driver/coyote_driver.ko
```


### Step 6: Run the DLRM training with FPGA preprocessing

Start the DLRM training benchmark configured to use the FPGA preprocessing pipeline and data transfer.

```bash
cd DLRM/dlrm/bench
bash dlrm_fpga_preproc_AMD.sh
```

## Compilation Instructions for RDMA Integration

Follow all steps described in the main README, replacing references to `11_preprocess_dlrm` with `12_preprocess_dlrm_rdma`, **except for Step 4 and Step 6**, which are modified as follows:

### Step 4: Compile the FPGA preprocessing software interfaces

#### Step 4.1: Compile the client interface
```bash
cd Coyote/examples/12_preprocess_dlrm_rdma/sw
mkdir -p build_client && cd build_client
export CXX=hipcc
cmake ../ -DEN_GPU=1 -DAMD_GPU=gfx90a -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make
```
This creates a shared library `.so` file in the build folder.

#### Step 4.2: Compile the server
```bash
cd Coyote/examples/12_preprocess_dlrm_rdma/sw
mkdir -p build_server && cd build_server
export CXX=hipcc
cmake ../ -DINSTANCE=server -DEN_GPU=1
make
```
This will build `test` (the RDMA server executable) inside `build_server/bin`.

### Step 6: Run the RDMA-enabled DLRM training with FPGA preprocessing
1. Start the server (on a separate node):
```bash
cd Coyote/examples/12_preprocess_dlrm_rdma/sw/build_server
bin/test
```

2. Run the DLRM training (client):
```bash
cd DLRM/dlrm/bench
bash dlrm_fpga_preproc_AMD.sh
```

**Note:** The experiments were performed on the ETH HACC cluster by running the server on the `hacc-box-01` node and the DLRM training on the `hacc-box-02` node. If using other machines, please update the server IP address in `Coyote/examples/12_preprocess_dlrm_rdma/sw/src/client/fpga_p2p_pybind.cpp`.

## Additional Information
To run the baseline CPU-GPU pipeline, follow the instructions mentioned in [`DLRM README`](DLRM/README.md).

## License

This repository contains code from two projects, each covered by their own MIT License:

- **Coyote framework components** — Copyright (c) 2022 FPGA @ Systems Group, ETH Zurich  
- **Meta's DLRM components** — Copyright (c) Facebook, Inc. and its affiliates  

The full license texts are provided in the [`LICENSE`](LICENSE.md) file.

