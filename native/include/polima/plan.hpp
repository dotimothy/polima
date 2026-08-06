// The execution-plan interpreter.
//
// This is what makes one binary serve every bundle. act_llima.cpp hard-codes
// ACT's pipeline in C++ -- six Runners named in a constructor initializer list,
// packing offsets as literals, a 16->6 unpad loop. Change the policy and you
// rebuild the binary on the board.
//
// Here that pipeline is plan.json:
//
//   run_elf        cam0   <- vision_backbone(image0)
//   run_elf        cam1   <- vision_backbone(image1)
//   pack           packed <- state@0, cam0@512, cam1@154112
//   run_elf        hidden <- encoder_layer_00_stem(packed)
//   run_elf        hidden <- encoder_layer_01(hidden)   [x3]
//   run_elf        padded <- decoder_action_tail(hidden)
//   gather_strided actions<- padded stride 16 take 6 count 100
//
// ACT needs exactly three opcodes. SmolVLA (Phase 4) adds six more; they are
// declared in the enum now so the dispatch is exhaustive from the start.
#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "polima/runner.hpp"
#include "polima/sidecar.hpp"

namespace polima {

enum class Opcode {
  RunElf,
  Pack,
  Slice,
  GatherStrided,
  Scale,
  MatVec,
  SincosTime,
  Euler,
  Normalize,
  Denormalize,
};

Opcode parse_opcode(const std::string& name);
const char* opcode_name(Opcode opcode);

struct PackPart {
  std::string source;
  size_t destination_offset = 0;
  size_t count = 0;
};

struct Step {
  Opcode opcode;
  std::string out;
  std::string graph;                 // run_elf
  std::string source;                // args["src"] -- slice, gather_strided, scale, ...
  std::vector<std::string> inputs;   // args["in"]  -- run_elf and friends
  std::vector<PackPart> parts;       // pack
  size_t offset = 0;                 // slice
  size_t stride = 0;                 // gather_strided
  size_t take = 0;
  size_t count = 0;
  size_t size = 0;
  float scalar = 0.0f;               // scale
  std::string weights;               // matvec / normalize
  std::string bias;
  std::string mean;
  std::string std_dev;
  size_t rows = 0;
  size_t cols = 0;
  std::string raw;                   // original JSON, for error messages
};

struct WireDescription {
  uint32_t magic = 0;
  uint32_t version = 1;
  int default_port = 0;
  std::vector<std::pair<std::string, size_t>> request_tensors;  // name -> elements
  size_t response_elements = 0;
};

// A bundle's plan.json plus the ELF inventory from bundle.json, loaded and
// wired into live Runners with every buffer allocated up front.
class Plan {
 public:
  Plan(const std::filesystem::path& bundle_root, bool verbose = false);

  // Runs the whole plan. `inputs` must supply every wire request tensor by name.
  const std::vector<float>& execute(const std::map<std::string, const float*>& inputs);

  // Run against whatever is already in the plan's own input buffers.
  // Callers that can write straight into `input_buffer()` should use this: it
  // avoids copying the request payload a second time, which for ACT is 7.4 MB
  // (two 480x640x3 images) on every single inference.
  const std::vector<float>& execute();

  // Writable pointer to a declared buffer, for filling inputs in place.
  float* input_buffer(const std::string& name);

  const WireDescription& wire() const { return wire_; }
  const std::string& bundle_id() const { return bundle_id_; }
  const std::string& policy() const { return policy_; }
  size_t result_elements() const { return buffer_sizes_.at(result_); }
  const std::vector<std::string>& stage_timings() const { return timings_; }

 private:
  std::vector<float>& buffer(const std::string& name);
  const std::vector<float>& buffer(const std::string& name) const;
  // Resolves a step's single read buffer: args["src"] if present, else in[0].
  const std::vector<float>& read_buffer(const Step& step) const;
  void run_step(const Step& step);

  std::filesystem::path root_;
  bool verbose_;
  std::string bundle_id_;
  std::string policy_;
  std::string result_;
  WireDescription wire_;

  std::map<std::string, std::vector<float>> buffers_;
  std::map<std::string, size_t> buffer_sizes_;
  std::map<std::string, std::unique_ptr<Runner>> runners_;
  std::vector<Step> steps_;
  Sidecars sidecars_;
  std::vector<std::string> timings_;

  std::unique_ptr<RuntimeGuard> guard_;
};

}  // namespace polima
