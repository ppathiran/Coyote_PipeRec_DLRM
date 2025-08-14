#  FPGA-GPU Training System for Recommender Models

This repository contains the implementation and integration of FPGA-based preprocessing with Meta's Deep Learning Recommendation Model (DLRM) using the Coyote framework.  
The goal of this project is to accelerate data preprocessing on FPGA and seamlessly transfer preprocessed data to the GPU for efficient DLRM training.

## Requirements

- **Hardware:**  
  - AMD GPU with HIP support (tested on gfx90a architecture)  
  - Xilinx U55C FPGA device  
- **Software:**  
  - AMD-compatible PyTorch (tested with PyTorch built for AMD GPUs)  
  - Linux environment (tested on the HACC cluster `hacc-box` machines)  
  - HIP development tools (hipcc compiler)  
  - CMake  
  - pybind11 Python bindings  
  - Python 3.10 or higher  


## Repository Structure

- `Coyote/` — Contains the FPGA preprocessing hardware design, drivers, and software interfaces based on the Coyote framework.  
- `DLRM/` — Contains the adapted DLRM codebase configured to use the FPGA-preprocessed data for training.


## Compilation and Setup Steps

All commands assume starting from the **root directory** of the repository.

### Step 1: Compile the FPGA bitstream

Compile the FPGA hardware preprocessing pipeline bitstream for the U55C device.  
This step synthesizes the FPGA design and generates the bitstream file.

```bash
cd Coyote/examples/09_preprocess/hw
mkdir build_hw && cd build_hw
cmake ../ -DFDEV_NAME=u55c
make project && make bitgen
```
The bitstream will be available in the build_hw/bitstreams/ folder.


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
cd Coyote/examples/11_preprocess_dlrm/sw/pointer_to_tensor
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH
python3 setup.py build_ext --inplace
```
This produces a .so shared object file inside the pointer_to_tensor folder.


### Step 4: Compile the FPGA preprocessing software interface

Build the C++/Python interface to run and manage FPGA preprocessing, using HIP for AMD GPU support.

```bash
cd Coyote/examples/11_preprocess_dlrm/sw
mkdir build && cd build
export CXX=hipcc
cmake ../ -DEN_GPU=1 -DAMD_GPU=gfx90a -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make
```
This creates a shared library .so file in the build folder.


### Step 5: Program the FPGA with the bitstream and driver

Load the generated bitstream onto the FPGA and insert the compiled driver module.

```bash
cd Coyote
bash util/program_hacc_local.sh examples/09_preprocess/hw/build_hw/bitstreams/cyt_top.bit driver/coyote_driver.ko
```


### Step 6: Run the DLRM training with FPGA preprocessing

Start the DLRM training benchmark configured to use the FPGA preprocessing pipeline and data transfer.

```bash
cd DLRM/dlrm/bench
bash dlrm_fpga_preproc_AMD.sh
```
