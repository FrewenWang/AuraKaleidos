# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
FantasyCXX (AuraKaleidos C++ Project) is a comprehensive C++ project collection focused on computer vision, performance monitoring, and configuration parsing. The project uses CMake as its build system and supports cross-platform compilation.

## Build Commands

### Primary Build Script
```bash
# Build for current platform (Linux/macOS)
./build.sh

# Build Android arm64-v8a
./build.sh -t 2 -d  # Debug version
./build.sh -t 2 -r  # Release version

# Build Android armeabi-v7a
./build.sh -t 1 -d  # Debug version
./build.sh -t 1 -r  # Release version
```

### Manual CMake Build
```bash
# Create build directory
mkdir -p build && cd build

# Configure project
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build with multiple threads
make -j$(nproc)

# Install
make install
```

### Build Options
- `BUILD_STATIC_LIB`: Build static libraries (default: OFF)
- `BUILD_EXAMPLES`: Build example code (default: ON)
- `BUILD_TOOLS`: Build tools (default: ON)
- `BUILD_UNIT_TEST`: Build unit tests (default: ON)
- `BUILD_OPENCV`: Enable OpenCV support (default: ON)
- `BUILD_GTEST_LIB`: Enable Google Test (default: ON)

## Testing Commands

### Build and Run Tests
```bash
# Build with tests enabled
cmake -DBUILD_UNIT_TEST=ON ..
make

# Run main test suite
cd test && ./run_tests

# Run specific module tests (examples)
cd aura-cv/src/test && ./aura-cv-tests
cd aura-utils/test && ./aura-utils-tests
```

### Test Organization
- Main test directory: `/test/`
- Module-specific tests: Each module contains its own test directory
- Test framework: Google Test (GTest)

## Code Style and Linting

### Clang Format
The project uses `.clang-format` for code formatting with LLVM-based style:
```bash
# Format code using project's clang-format configuration
clang-format -i source_file.cpp
clang-format -i header_file.h
```

### Key Style Points
- C++17 standard compliance
- 4-space indentation (no tabs)
- Right-aligned pointers and references
- Custom brace wrapping rules
- Access modifier offset of -4

## Project Architecture

### Core Module Dependencies
```
Main CMakeLists.txt
├── aura-utils (foundation utilities)
├── aura-cv (core computer vision)
├── samples (demonstration code)
├── aura-perfguard (performance monitoring)
└── Conditional modules (commented out):
    ├── aura-auto-driving (requires Eigen)
    ├── test (requires GTest)
    ├── aura-lightbuffer (header issues)
    ├── aura-vision (header issues)
    └── aura-vision-hpc (OpenCL issues)
```

### Module Structure
Each module follows a consistent pattern:
- `CMakeLists.txt`: Module build configuration
- `src/`: Source code implementation
- `include/`: Public header files (if applicable)
- `test/`: Unit tests
- `samples/` or `examples/`: Demonstration code

### Third-party Dependencies
The project integrates multiple third-party libraries:
- **OpenCV**: Computer vision operations
- **GTest**: Unit testing framework
- **Eigen**: Linear algebra operations
- **Ceres**: Nonlinear optimization
- **Boost**: C++ extensions
- **JSON**: Configuration parsing
- **OSQP**: Optimization solver

## Platform Support Matrix

| Platform | Architecture | Build Target | Notes |
|----------|-------------|--------------|-------|
| Linux | x86_64 | `linux-x86_64` | Primary development platform |
| macOS | x86_64 | `mac-x86_64` | Primary development platform |
| Android | arm64-v8a | `android-arm64-v8a` | Requires NDK |
| Android | armeabi-v7a | `android-armeabi-v7a` | Requires NDK |
| Windows | x86_64 | `windows-x86_64` | Limited support |

## Development Workflow

### Adding New Modules
1. Create module directory in project root
2. Implement `CMakeLists.txt` following existing patterns
3. Add source code in `src/` subdirectory
4. Add public headers in `include/` if needed
5. Create tests in `test/` subdirectory
6. Add module to main `CMakeLists.txt` with `add_subdirectory()`

### Working with Examples
- Main samples: `/samples/` directory
- Module-specific examples: Located within each module
- Build examples: Enable with `BUILD_EXAMPLES=ON`

### Cross-Platform Development
- Use CMake toolchain files for cross-compilation
- Android builds require NDK_HOME environment variable
- Platform-specific code should use CMake conditionals

## Configuration Parameters

### Build-time CMake Variables
- `TARGET_OS`: Target operating system (android/linux/mac/windows)
- `TARGET_ARCH`: Target architecture (x86_64/arm64-v8a/armeabi-v7a)
- `CMAKE_BUILD_TYPE`: Build type (debug/release/RelWithDebInfo)
- `PRODUCTION`: Product name identifier
- `SOC_VENDOR`: SoC vendor (e.g., qcom)

### Environment Variables
- `NDK_HOME`: Android NDK path (required for Android builds)
- `CMAKE_INSTALL_PREFIX`: Installation directory

## Key Files and Directories

### Build System
- `CMakeLists.txt`: Main build configuration
- `build.sh`: Automated build script
- `cmake/`: CMake modules and utilities
- `3rdparty/`: Third-party dependency management

### Configuration
- `.clang-format`: Code formatting rules
- `README.md`: Project documentation

### Source Organization
- `aura-cv/`: Core computer vision algorithms
- `aura-utils/`: Foundation utilities and helpers
- `samples/`: Demonstration and example code
- `test/`: Main test suite

## Troubleshooting

### Common Build Issues
1. **CMake version**: Ensure CMake 3.10+ is installed
2. **Missing dependencies**: Check third-party libraries are available
3. **Android builds**: Verify NDK_HOME environment variable is set
4. **Cross-compilation**: Ensure proper toolchain configuration

### Module-specific Issues
- Some modules are disabled in main CMakeLists.txt due to dependency issues
- Check module-specific README files for additional requirements
- Verify platform compatibility before enabling modules

## Recent Development Focus
Based on git history, recent work emphasizes:
- Documentation improvements (README updates)
- New demonstrations and examples
- CUDA integration
- Code commenting and documentation
- Third-party library integration and testing