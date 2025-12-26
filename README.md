# 3D PCC Simulation for Pneumatic Soft-Robotics Actuators

A real-time 3D physics-enabled simulation framework for pneumatic soft robotics with CUDA acceleration, featuring forward and inverse kinematics modeling.

![Project Banner](https://img.shields.io/badge/Soft_Robotics-Simulation-blue)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green)


## 🎯 Overview

This project presents a comprehensive 3D physics-based simulation framework for pneumatic soft robotic actuators using the Piecewise Constant Curvature (PCC) model. The framework includes:

- **GPU-Accelerated Forward Kinematics**: Real-time computation using CUDA
- **Inverse Kinematics**: CMA-ES optimization for target position reaching
- **Interactive GUI**: PySide6-based interface with 3D visualization
- **Material Library**: Support for multiple hyperelastic material models
- **Physics Validation**: Built-in checks for material properties and geometric constraints

## ✨ Features

### Core Capabilities
- ⚡ **Real-time CUDA acceleration** for forward kinematics computation
- 🎯 **CMA-ES based inverse kinematics** for target position optimization
- 🎨 **3D visualization** with PyVista for workspace analysis
- 📊 **Material library management** for various soft materials
- 🔬 **Physics-based modeling** using hyperelastic material models
- ⚙️ **Multi-segment support** with independent geometric parameters

### Material Models
- **Neo-Hookean**: Simple hyperelastic model
- **Mooney-Rivlin**: Two-parameter hyperelastic model
- **Ogden**: General hyperelastic model with power-law behavior

### Validation Features
- Incompressibility checks
- Thin-wall assumption verification
- Self-collision detection
- Geometric constraint validation

## 🖥️ System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with compute capability 8.9 (Ada Lovelace architecture)
  - Alternatively, modify `CMakeLists.txt` for your GPU architecture
- **RAM**: Minimum 8GB recommended
- **Storage**: ~500MB for project and dependencies

### Software Requirements
- **Operating System**: Linux (Ubuntu 24 recommended)
- **CUDA Toolkit**: Version 12.6 (or compatible)
- **CMake**: Version 3.18 or higher
- **Python**: Version 3.8 or higher
- **C++ Compiler**: Support for C++20 standard (g++ 11+ or clang 14+)

### Python Dependencies
- PySide6 >= 6.0.0 (Qt for Python)
- pyvista >= 0.43.0 (3D visualization)
- pyvistaqt >= 0.11.0 (PyVista Qt integration)
- numpy >= 1.21.0 (numerical computing)
- cma >= 3.3.0 (CMA-ES optimization)

## 🚀 Installation


#### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Install Python Dependencies
```bash
pip install --upgrade pip
pip install PySide6 pyvista pyvistaqt numpy cma
```

#### Step 3: Build CUDA Library
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

#### Step 4: Verify Installation
```bash
# Check if the shared library was created
ls -la lib/fk.so
```

## 🎬 Quick Start

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

## 📁 Project Structure

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


## ⚙️ Configuration

### GPU Architecture

If you have a different NVIDIA GPU, modify `CMakeLists.txt`:

```cmake
# Change this line to match your GPU compute capability
set(CMAKE_CUDA_ARCHITECTURES 89)  # 8.9 for Ada Lovelace

# Update compile options
-gencode=arch=compute_89,code=sm_89  # Change both instances
```

Common compute capabilities:
- 7.5: Turing (RTX 2000 series)
- 8.0: Ampere (A100)
- 8.6: Ampere (RTX 3000 series)
- 8.9: Ada Lovelace (RTX 4000 series)
- 9.0: Hopper (H100)

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

## 📚 Additional Resources

- **Technical Report**: See `docs/Submission_report.pdf` for detailed mathematical formulation
- **Installation Guide**: See `installation.ipynb` for step-by-step setup
- **Source Code**: [GitHub Repository](https://github.com/adityawardhanm/3d-pcc-simulation)

## 🙏 Acknowledgments

Special thanks to:
- Prof. Jahan Gul and Maynooth University for guidance and support
- The soft robotics research community
- CUDA and PyVista development teams

## 👤 Author

**Adityawardhan Mishra**  
National University of Ireland, Maynooth