cmake_minimum_required(VERSION 3.18)
project(cuda_cpp_project LANGUAGES C CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include headers
include_directories(${PROJECT_SOURCE_DIR}/include)

SET(SOURCES
    src/matrice_gpu.cpp
    src/Network.cpp
)

# Build the CUDA library
add_library(matrice_helper STATIC
    src/matrice_helper.cu
)

# Mark that this target contains CUDA code
set_target_properties(matrice_helper PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Build the executable
add_executable(main
    main.cpp
)

# Link the CUDA library
target_link_libraries(main PRIVATE matrice_helper)
