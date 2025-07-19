cmake_minimum_required(VERSION 3.22)
project(DVAttnCache CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89")

find_package(CUDAToolkit REQUIRED)

option(DVATTN_TORCH_BINDINGS "Build pytorch bindings" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()


add_subdirectory(src/tpl)

option(DVATTN_ENABLE_FLOAT "build with float32 kernel support" ON)
option(DVATTN_ENABLE_HALF "build with float16 kernel support" ON)
option(DVATTN_ENABLE_BFLOAT "build with bfloat16 kernel support" ON)

add_library(dvattn-kernels STATIC ${DVATTN_KERNEL_INSTANTIATIONS})
target_include_directories(dvattn-kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(dvattn-kernels PUBLIC -expt-relaxed-constexpr -lineinfo)

if(DVATTN_ENABLE_FLOAT)
    target_compile_definitions(dvattn-kernels PUBLIC DVATTN_ENABLE_FLOAT=1)
endif ()

if(DVATTN_ENABLE_HALF)
    target_compile_definitions(dvattn-kernels PUBLIC DVATTN_ENABLE_HALF=1)
endif ()

if(DVATTN_ENABLE_BFLOAT)
    target_compile_definitions(dvattn-kernels PUBLIC DVATTN_ENABLE_BFLOAT=1)
endif ()


if(DVATTN_TORCH_BINDINGS)
    include(cmake/pytorch.cmake)
    find_package(Python3 COMPONENTS Development.SABIModule REQUIRED)
    Python3_add_library(dvatt MODULE src/binding.cu WITH_SOABI USE_SABI 3.10)
    if(SKBUILD)
        install(TARGETS dvatt DESTINATION ${SKBUILD_PLATLIB_DIR}/dvattn)
    endif ()
    target_link_libraries(dvatt PRIVATE torch dvattn-kernels)
    target_compile_options(dvatt PUBLIC -expt-relaxed-constexpr -lineinfo)
    target_include_directories(dvatt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif ()
