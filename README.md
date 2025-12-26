# 3D PCC Simulation for Pneumatic Soft-Robotics Actuators

![Python](https://img.shields.io/badge/Python-3.10.12-3776AB?style=for-the-badge&logo=python&logoColor=white)&nbsp;![CUDA](https://img.shields.io/badge/CUDA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)&nbsp;

## About

This project was a project submission for a Soft Robotics Course I attended at Maynooth University during my MSc in Robotics and Embedded A.I. 

The project includes the a Forward and Inverse Kinematics Pipeline to simulate the movement of a 3d soft bellow actuator. The actuator in this case is a 4-channel pneumatic actuator and we use Piecewise Constant Curvature Model to simulate it.

## Features 

* **Forward Kinematics** - We use a a multi stage physics pipeline to simulate the robot for multiple segments. 
* **Inverse Kinematics** - CMA-ES optimization for target position reaching. 
* **CUDA Acceleration** - A CUDA-acceleration is used to solve the forward kinematics spline generation. (P.S. The spline does start clipping into itself)
* **Material Library** - The project using multiple materials with a json registry. 
* **Interative GUI** - A PySide6 (Python QT) + PyVista interface for 3d visualisation

## System Requirements

### Hardware Requirements
- **GPU** - An Nvidia GPU with compute capability 8.9 (Change `CMakeLists.txt` for your gpu architecture)
- **RAM** - Minimum 8GB recommended
- **Storage** - ~500MB for project and dependencies

### Software Requirements
- **Operating System** - Linux (Ubuntu 22 recommended)
- **CUDA Toolkit** - Version 12.6 (or compatible)
- **CMake** - Version 3.18 or higher
- **Python** - Version 3.10 or higher
- **C++ Compiler** - Support for C++20 standard

### Python Dependencies
- **PySide6** >= 6.10.1 (Qt for Python)
- **pyvista** >= 0.46.4 (3D visualization)
- **pyvistaqt** >= 0.11.3 (PyVista Qt integration)
- **numpy** >= 1.21.0 (numerical computing)
- **cma** >= 4.4.1 (CMA-ES optimization)

## Installation


* **Step 1**: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

* **Step 2**: Install Python Dependencies
```bash
pip install --upgrade pip
pip install PySide6 pyvista pyvistaqt numpy cma
```

* **Step 3**: Build CUDA Library
```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make

# Return to project root
cd ..
```

* **Step 4**: Verify Installation
```bash
# Check if the shared library was created
ls -la lib/fk.so
```

## Quick Start

### Running the Application

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run the main application
python3 ./gui/main.py
```

### Basic Usage Example

1. **Launch the GUI**: Run `python main.py` (present in /gui folder)
2. **Configure Robot**:
   - Set number of segments
   - Define segment dimensions (length, radius, thickness)
   - Select material model
3. **Control Pressures**:
   - Use sliders to adjust channel pressures (pA, pB, pC, pD)
   - Observe real-time 3D visualization
4. **Inverse Kinematics**:
   - Input target position (X, Y, Z)
   - Click "Solve IK" to find optimal pressures
5. **Workspace Analysis**:
   - Generate workspace point clouds
   - Visualize reachable positions

## Project Structure

```
3d-pcc-simulation/
├── README.md                       # This file
├── CMakeLists.txt                  # CMake configuration
├── gui/
│   ├── main.py                     # GUI Entry
│   └── widgets.py                  # GUI Components
│ 
├── src/
│   ├── python/                     # Python source files
│   │   ├── widgets.py              # GUI implementation
│   │   ├── fk.py                   # Forward kinematics
│   │   ├── ik.py                   # Inverse kinematics
│   │   ├── spline.py               # Spline generation
│   │   ├── checks.py               # Physics validation
│   │   └── materials_library.py    # Material management
│   │
│   └── cuda/                       # CUDA source files
│       └── fk.cu                   # CUDA kernels for FK
│
├── lib/                            # Compiled libraries
│   └── fk.so                       # CUDA shared library (generated)
│
├── docs/                           # Documentation
│   └── Submission_report.pdf       # Technical report
│
├── material_library.json           # Material presets
├── default_materials.json          # Default materials
└── installation.ipynb              # Installation
```


## Configuration

### GPU Architecture

If you have a different NVIDIA GPU, modify `CMakeLists.txt`:

```cmake
# Change this line to match your GPU compute capability
set(CMAKE_CUDA_ARCHITECTURES 89)  # 8.9 for Ada Lovelace

# Update compile options
-gencode=arch=compute_89,code=sm_89  # Change both instances
```


### CUDA Toolkit Path

If CMake cannot find CUDA, update the path in `CMakeLists.txt`:

```cmake
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
# Change to your CUDA installation path
```

### Material Library

Edit `default_materials.json` to customize default materials:

```json
{
  "Ecoflex 00-50": {
    "epsilon_pre": 0.15,
    "bulk_modulus": 50000000
  },
  "Custom Material": {
    "epsilon_pre": 0.20,
    "bulk_modulus": 100000000
  }
}
```

## Acknowledgments

Special thanks to:
- Prof. Jahan Gul and Maynooth University for guidance and support
- The soft robotics research community
- CUDA and PyVista development teams

## License 

This project is licensed under the [MIT License](LICENSE).

## Support

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-%23FF813F?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://www.buymeacoffee.com/adityawardhanm)