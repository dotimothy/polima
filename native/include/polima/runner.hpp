// One compiled graph on the MLA.
//
// Generalizes the `Runner` struct that act_llima.cpp, smolvla_som_server.cpp and
// smolvla_action_llima.cpp each define separately. The difference here is that
// shapes come from bundle.json rather than being baked into a constructor
// initializer list (act_llima.cpp:117-122), which is what lets one binary serve
// every bundle.
#pragma once

#include <sima_lmm/mla_buffer.hpp>
#include <sima_lmm/mla_model.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include "polima/bf16.hpp"
#include "polima/tess.hpp"

namespace polima {

namespace llima = simaai::llima;

// Connects the MLA runtime for the process lifetime. One per process.
struct RuntimeGuard {
  RuntimeGuard() { llima::connect_mla_rt({}); }
  ~RuntimeGuard() { llima::disconnect_mla_rt(); }
  RuntimeGuard(const RuntimeGuard&) = delete;
  RuntimeGuard& operator=(const RuntimeGuard&) = delete;
};

class Runner {
 public:
  Runner(std::string name, const std::filesystem::path& elf, size_t input_elements,
         size_t output_elements, DramLayout output_layout = DramLayout::Plain,
         size_t logical_width = 0, size_t logical_channels = 0)
      : name_(std::move(name)),
        input_elements_(input_elements),
        output_elements_(output_elements),
        output_layout_(output_layout),
        logical_width_(logical_width),
        logical_channels_(logical_channels),
        staged_input_(input_elements),
        staged_output_(output_elements),
        input_(name_ + "_ifm", {1, input_elements}, "bfloat16", false),
        output_(name_ + "_ofm", {1, output_elements}, "bfloat16", false),
        model_(elf, {llima::MLABufferSlice(&input_)}, {llima::MLABufferSlice(&output_)}) {
    input_.allocate();
    output_.allocate();
    input_.clear();
    output_.clear();
    model_.load();
  }

  ~Runner() {
    model_.free();
    input_.free();
    output_.free();
  }

  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  // No allocation on this path: the staging vectors are sized once in the
  // constructor. The legacy Runner::run allocated two vectors per call, on
  // every request, for every stage.
  void run(const float* input, float* output) {
    to_bf16_span(input, staged_input_.data(), input_elements_);
    input_.upload(staged_input_.data());
    model_.run();
    output_.download(staged_output_.data());
    detessellate(output_layout_, staged_output_.data(), output_elements_, logical_width_,
                 logical_channels_, output);
  }

  const std::string& name() const { return name_; }
  size_t input_elements() const { return input_elements_; }
  size_t output_elements() const { return output_elements_; }

 private:
  std::string name_;
  size_t input_elements_;
  size_t output_elements_;
  DramLayout output_layout_;
  size_t logical_width_;
  size_t logical_channels_;
  std::vector<uint16_t> staged_input_;
  std::vector<uint16_t> staged_output_;
  llima::MLABuffer input_;
  llima::MLABuffer output_;
  llima::MLAModelWithBuffer model_;
};

}  // namespace polima
