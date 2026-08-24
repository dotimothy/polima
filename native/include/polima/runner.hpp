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

#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "polima/bf16.hpp"
#include "polima/tess.hpp"

namespace polima {

namespace llima = simaai::llima;

// The MLA runtime connection, shared by every Plan in the process.
//
// It used to be one guard per Plan, which was wrong the moment a second Plan
// could exist: two connects, and then the FIRST plan destroyed would
// disconnect the runtime out from under the one still using it. The interactive
// session makes that reachable -- `use` a second model and the first one's
// teardown pulls the runtime.
//
// So it is refcounted: connect on the first Plan, disconnect when the last one
// goes away. `acquire_mla_runtime()` is the only way to get one.
struct RuntimeGuard {
  RuntimeGuard() { llima::connect_mla_rt({}); }
  ~RuntimeGuard() { llima::disconnect_mla_rt(); }
  RuntimeGuard(const RuntimeGuard&) = delete;
  RuntimeGuard& operator=(const RuntimeGuard&) = delete;
};

std::shared_ptr<RuntimeGuard> acquire_mla_runtime();

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

struct SharedStage {
  std::string name;
  std::filesystem::path elf;
  size_t input_elements = 0;
  size_t output_elements = 0;
};

// A linear sequence of HWC-compiled ELFs with device-resident intermediates.
// Only the first input is uploaded and only the final output is downloaded.
// Intermediate stages alternate between two MLABuffers, matching the
// ping-pong arrangement validated for GR00T EAGLE on ModaliX.
class SharedRunnerChain {
 public:
  SharedRunnerChain(std::string name, const std::vector<SharedStage>& stages,
                    DramLayout output_layout = DramLayout::Plain,
                    size_t logical_width = 0, size_t logical_channels = 0)
      : name_(std::move(name)), output_layout_(output_layout),
        logical_width_(logical_width), logical_channels_(logical_channels) {
    if (stages.size() < 2)
      throw std::runtime_error(name_ + " shared chain needs at least two stages");
    input_elements_ = stages.front().input_elements;
    output_elements_ = stages.back().output_elements;
    const size_t shared_elements = stages.front().output_elements;
    for (size_t index = 1; index < stages.size(); ++index) {
      if (stages[index].input_elements != shared_elements ||
          stages[index].output_elements != shared_elements)
        throw std::runtime_error(name_ + " stage " + stages[index].name +
                                 " does not match the shared buffer size");
    }
    if (output_elements_ != shared_elements)
      throw std::runtime_error(name_ + " final output does not match its shared buffer");

    staged_input_.resize(input_elements_);
    staged_output_.resize(output_elements_);
    input_ = std::make_unique<DeviceBuffer>(name_ + "_ifm", input_elements_);
    ping_ = std::make_unique<DeviceBuffer>(name_ + "_ping", shared_elements);
    pong_ = std::make_unique<DeviceBuffer>(name_ + "_pong", shared_elements);
    for (size_t index = 0; index < stages.size(); ++index) {
      auto* source = index == 0 ? input_.get() : (index % 2 ? ping_.get() : pong_.get());
      auto* destination = index % 2 ? pong_.get() : ping_.get();
      models_.push_back(std::make_unique<llima::MLAModelWithBuffer>(
          stages[index].elf, std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(&source->storage)},
          std::vector<llima::MLABufferSlice>{llima::MLABufferSlice(&destination->storage)}));
      models_.back()->load();
    }
    final_ = stages.size() % 2 ? ping_.get() : pong_.get();
  }

  ~SharedRunnerChain() {
    for (auto& model : models_) model->free();
    models_.clear();
  }

  SharedRunnerChain(const SharedRunnerChain&) = delete;
  SharedRunnerChain& operator=(const SharedRunnerChain&) = delete;

  void run(const float* input, float* output) {
    to_bf16_span(input, staged_input_.data(), input_elements_);
    input_->storage.upload(staged_input_.data());
    for (auto& model : models_) model->run();
    final_->storage.download(staged_output_.data());
    detessellate(output_layout_, staged_output_.data(), output_elements_, logical_width_,
                 logical_channels_, output);
  }

  size_t input_elements() const { return input_elements_; }
  size_t output_elements() const { return output_elements_; }

 private:
  struct DeviceBuffer {
    DeviceBuffer(const std::string& name, size_t elements)
        : storage(name, {1, elements}, "bfloat16", false) {
      storage.allocate();
      storage.clear();
    }
    ~DeviceBuffer() { storage.free(); }
    llima::MLABuffer storage;
  };

  std::string name_;
  size_t input_elements_ = 0;
  size_t output_elements_ = 0;
  DramLayout output_layout_;
  size_t logical_width_;
  size_t logical_channels_;
  std::vector<uint16_t> staged_input_;
  std::vector<uint16_t> staged_output_;
  std::unique_ptr<DeviceBuffer> input_;
  std::unique_ptr<DeviceBuffer> ping_;
  std::unique_ptr<DeviceBuffer> pong_;
  DeviceBuffer* final_ = nullptr;
  std::vector<std::unique_ptr<llima::MLAModelWithBuffer>> models_;
};

}  // namespace polima
