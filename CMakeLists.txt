cmake_minimum_required(VERSION 3.19...3.30)
project(intextus_backend LANGUAGES CXX)

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

# FORCE PIC FOR ALL DEPENDENCIES (Fixes the R_X86_64_PC32 linker error)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# 1. Fetch tokenizers-cpp
FetchContent_Declare(
    tokenizers-cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        main
)

FetchContent_GetProperties(tokenizers-cpp)
if(NOT tokenizers-cpp_POPULATED)
    FetchContent_Populate(tokenizers-cpp)
    
    # Patch tokenizers-cpp/CMakeLists.txt to not hardcode aarch64-unknown-linux-gnu cross-compilers
    # since we are compiling inside native aarch64 docker containers under QEMU emulation.
    set(TC_CMAKE_FILE "${tokenizers-cpp_SOURCE_DIR}/CMakeLists.txt")
    if(EXISTS "${TC_CMAKE_FILE}")
        file(READ "${TC_CMAKE_FILE}" TC_CMAKE_CONTENT)
        string(REPLACE
            "AR_\${TOKENIZERS_CPP_CARGO_TARGET}=\${TOOLCHAIN_DIR}\${TOKENIZERS_CPP_CARGO_TARGET}-ar"
            "AR_\${TOKENIZERS_CPP_CARGO_TARGET}=ar"
            TC_CMAKE_CONTENT "${TC_CMAKE_CONTENT}")
        string(REPLACE
            "CC_\${TOKENIZERS_CPP_CARGO_TARGET}=\${TOOLCHAIN_DIR}\${TOKENIZERS_CPP_CARGO_TARGET}-gcc"
            "CC_\${TOKENIZERS_CPP_CARGO_TARGET}=gcc"
            TC_CMAKE_CONTENT "${TC_CMAKE_CONTENT}")
        string(REPLACE
            "CXX_\${TOKENIZERS_CPP_CARGO_TARGET}=\${TOOLCHAIN_DIR}\${TOKENIZERS_CPP_CARGO_TARGET}-g++"
            "CXX_\${TOKENIZERS_CPP_CARGO_TARGET}=g++"
            TC_CMAKE_CONTENT "${TC_CMAKE_CONTENT}")
        file(WRITE "${TC_CMAKE_FILE}" "${TC_CMAKE_CONTENT}")
    endif()

    add_subdirectory(${tokenizers-cpp_SOURCE_DIR} ${tokenizers-cpp_BINARY_DIR})
endif()

# 2. Fetch ONNX Runtime precompiled binaries and headers
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(ORT_ARCH "osx-arm64")
    else()
        set(ORT_ARCH "osx-x86_64")
    endif()
    set(ORT_SUFFIX "tgz")
elseif(WIN32)
    set(ORT_ARCH "win-x64")
    set(ORT_SUFFIX "zip")
else()
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(ORT_ARCH "linux-aarch64")
    else()
        set(ORT_ARCH "linux-x64")
    endif()
    set(ORT_SUFFIX "tgz")
endif()

set(ORT_VERSION "1.20.1")
set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_ARCH}-${ORT_VERSION}.${ORT_SUFFIX}")

FetchContent_Declare(
    onnxruntime_binaries
    URL ${ORT_URL}
)
FetchContent_MakeAvailable(onnxruntime_binaries)

set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_binaries_SOURCE_DIR}/include")
if(WIN32)
    set(ONNXRUNTIME_LIBRARY "${onnxruntime_binaries_SOURCE_DIR}/lib/onnxruntime.lib")
elseif(APPLE)
    set(ONNXRUNTIME_LIBRARY "${onnxruntime_binaries_SOURCE_DIR}/lib/libonnxruntime.dylib")
else()
    set(ONNXRUNTIME_LIBRARY "${onnxruntime_binaries_SOURCE_DIR}/lib/libonnxruntime.so")
endif()

# 3. Locate nanobind automatically using your current Python environment
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind REQUIRED)

# 4. Define our C++ extension module
nanobind_add_module(
    _core                         # The name of the module
    src/intextus/_core.cpp        # Where the C++ source code is
)

# 5. Add include paths and link libraries
target_include_directories(_core PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(_core PRIVATE tokenizers_cpp ${ONNXRUNTIME_LIBRARY})

# Configure RPATH for the extension module to find libraries in the same directory
if(APPLE)
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(_core PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# 6. Tell the build system to drop the finished binary and dependencies inside your python package folder
install(TARGETS _core DESTINATION intextus)

if(WIN32)
    file(GLOB ORT_LIBS "${onnxruntime_binaries_SOURCE_DIR}/lib/onnxruntime.dll")
    install(FILES ${ORT_LIBS} DESTINATION intextus)
elseif(APPLE)
    file(GLOB ORT_LIBS "${onnxruntime_binaries_SOURCE_DIR}/lib/libonnxruntime*.dylib")
    install(FILES ${ORT_LIBS} DESTINATION intextus)
else()
    file(GLOB ORT_LIBS "${onnxruntime_binaries_SOURCE_DIR}/lib/libonnxruntime.so*")
    install(FILES ${ORT_LIBS} DESTINATION intextus)
endif()