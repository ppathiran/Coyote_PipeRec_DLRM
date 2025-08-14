# Coyote_PipeRec_DLRM

This repository contains the implementation and integration of FPGA-based preprocessing with Meta's Deep Learning Recommendation Model (DLRM) using the Coyote framework.  
The goal of this project is to accelerate data preprocessing on FPGA and seamlessly transfer preprocessed data to the GPU for efficient DLRM training.

---

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

---

## Repository Structure

- `Coyote/` — Contains the FPGA preprocessing hardware design, drivers, and software interfaces based on the Coyote framework.  
- `DLRM/` — Contains the adapted DLRM codebase configured to use the FPGA-preprocessed data for training.

---

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
