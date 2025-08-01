// This file is automatically generated.
// Do not edit by hand!

#pragma once

#include "common.h"
#include <array>
#include <cstdio>
#include <string>

\
    #define DECL_KERNEL(ns)\
    namespace ns {\
    template<class scalar_t>\
cudaError_t dvattn_attention_gpu(\
        scalar_t* out, float scale,\
        const int* locations, const scalar_t* queries,\
        const int* fragment_lengths,\
        const scalar_t** key_fragments,\
        const scalar_t** value_fragments,\
        Shape shape);}

DECL_KERNEL(ernel)
DECL_KERNEL(v21)
#undef DECL_KERNEL

template <class scalar_t>
cudaError_t dvattn_attention_gpu_dispatch(
    scalar_t *out, float scale, const int *locations, const scalar_t *queries,
    const int *fragment_lengths, const scalar_t **key_fragments,
    const scalar_t **value_fragments, const Shape &shape,
    const std::string &version) {
  if (version == "ernel") {
    return ernel::dvattn_attention_gpu(out, scale, locations, queries,
                                       fragment_lengths, key_fragments,
                                       value_fragments, shape);
  } else if (version == "v21") {
    return v21::dvattn_attention_gpu(out, scale, locations, queries,
                                     fragment_lengths, key_fragments,
                                     value_fragments, shape);
  } else {
    fprintf(stderr, "Invalid kernel version `%s`!", version.c_str());
    std::exit(1);
  }
}
constexpr const int NUM_KERNEL_VERSIONS = 2;
const std::array<std::string, NUM_KERNEL_VERSIONS> &get_all_versions() {
  static std::array<std::string, NUM_KERNEL_VERSIONS> versions = {"ernel",
                                                                  "v21"};
  return versions;
}
