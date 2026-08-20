#include "polima/plan.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace polima {

std::shared_ptr<RuntimeGuard> acquire_mla_runtime() {
  // Connect once for the process and disconnect when the last Plan is gone.
  // The weak_ptr is what makes the second half work: a released Plan drops the
  // refcount, and only a drop to zero tears the runtime down.
  static std::mutex lock;
  static std::weak_ptr<RuntimeGuard> shared;
  const std::lock_guard<std::mutex> held(lock);
  if (auto existing = shared.lock()) return existing;
  auto created = std::make_shared<RuntimeGuard>();
  shared = created;
  return created;
}

namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

json read_json(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  json value;
  stream >> value;
  return value;
}

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void require_finite(const std::string& stage, const std::vector<float>& values) {
  for (size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index]))
      throw std::runtime_error(stage + " produced a non-finite value at index " +
                               std::to_string(index));
  }
}

struct GraphInfo {
  std::string name;
  std::filesystem::path elf;
  size_t inputs = 0;
  size_t outputs = 0;
  DramLayout output_layout = DramLayout::Plain;
  size_t logical_width = 0;
  size_t logical_channels = 0;
  std::string external_layout = "compiler";
};

}  // namespace

Opcode parse_opcode(const std::string& name) {
  if (name == "run_elf") return Opcode::RunElf;
  if (name == "run_elf_chain") return Opcode::RunElfChain;
  if (name == "pack") return Opcode::Pack;
  if (name == "slice") return Opcode::Slice;
  if (name == "gather_strided") return Opcode::GatherStrided;
  if (name == "pixel_unshuffle") return Opcode::PixelUnshuffle;
  if (name == "scale") return Opcode::Scale;
  if (name == "matvec") return Opcode::MatVec;
  if (name == "sincos_time") return Opcode::SincosTime;
  if (name == "euler") return Opcode::Euler;
  if (name == "normalize") return Opcode::Normalize;
  if (name == "denormalize") return Opcode::Denormalize;
  throw std::runtime_error("unknown opcode '" + name + "' in plan.json");
}

const char* opcode_name(Opcode opcode) {
  switch (opcode) {
    case Opcode::RunElf: return "run_elf";
    case Opcode::RunElfChain: return "run_elf_chain";
    case Opcode::Pack: return "pack";
    case Opcode::Slice: return "slice";
    case Opcode::GatherStrided: return "gather_strided";
    case Opcode::PixelUnshuffle: return "pixel_unshuffle";
    case Opcode::Scale: return "scale";
    case Opcode::MatVec: return "matvec";
    case Opcode::SincosTime: return "sincos_time";
    case Opcode::Euler: return "euler";
    case Opcode::Normalize: return "normalize";
    case Opcode::Denormalize: return "denormalize";
  }
  return "?";
}

Plan::Plan(const std::filesystem::path& bundle_root, bool verbose)
    : root_(std::filesystem::canonical(bundle_root)), verbose_(verbose),
      collect_timings_(verbose) {
  const json manifest = read_json(root_ / "bundle.json");
  const json plan = read_json(root_ / "plan.json");

  bundle_id_ = manifest.value("bundle_id", "");
  policy_ = manifest.value("policy", "");
  if (manifest.contains("smoke")) {
    const json& smoke = manifest.at("smoke");
    smoke_cosine_min_ = smoke.value("cosine_min", smoke_cosine_min_);
    smoke_mean_abs_max_ = smoke.value("mean_abs_max", smoke_mean_abs_max_);
  }
  result_ = plan.at("result").get<std::string>();

  // Allocate every buffer once, at load time. Nothing on the request path
  // resizes or reallocates.
  for (auto item = plan.at("buffers").begin(); item != plan.at("buffers").end(); ++item) {
    const auto size = item.value().get<size_t>();
    buffer_sizes_[item.key()] = size;
    buffers_[item.key()] = std::vector<float>(size, 0.0f);
  }
  if (!buffer_sizes_.count(result_))
    throw std::runtime_error("plan result '" + result_ + "' is not a declared buffer");

  if (plan.contains("wire")) {
    const json& wire = plan.at("wire");
    wire_.magic = wire.value("magic", 0u);
    wire_.version = wire.value("version", 1u);
    wire_.default_port = wire.value("default_port", 0);
    wire_.response_elements = wire.value("response_elements", size_t{0});
    for (const auto& tensor : wire.value("request_tensors", json::array()))
      wire_.request_tensors.emplace_back(tensor.at("name").get<std::string>(),
                                         tensor.at("elements").get<size_t>());
  }

  sidecars_ = Sidecars(root_ / "constants");

  std::set<std::string> chained_graphs;
  std::set<std::string> regular_graphs;
  for (const auto& entry : plan.at("steps")) {
    const auto op = entry.at("op").get<std::string>();
    const auto args = entry.value("args", json::object());
    if (op == "run_elf") regular_graphs.insert(args.at("graph").get<std::string>());
    if (op == "run_elf_chain")
      for (const auto& graph : args.at("graphs"))
        chained_graphs.insert(graph.get<std::string>());
  }
  for (const auto& name : chained_graphs)
    if (regular_graphs.count(name))
      throw std::runtime_error("graph '" + name +
                               "' cannot be used both directly and in a shared chain");

  // Graph metadata drives both ordinary Runners and shared chains. Chained
  // graphs are not also loaded as ordinary Runners: GR00T has 25 of them and
  // loading every ELF twice would waste substantial device memory.
  guard_ = acquire_mla_runtime();
  std::map<std::string, GraphInfo> graph_infos;
  for (const auto& graph : manifest.at("graphs")) {
    const auto name = graph.at("name").get<std::string>();
    const auto elf = root_ / graph.at("elf").get<std::string>();
    if (!std::filesystem::exists(elf))
      throw std::runtime_error("bundle.json names a missing ELF: " + elf.string());

    GraphInfo info{name, elf, graph.at("input_elements").get<size_t>(),
                   graph.at("output_elements").get<size_t>(),
                   parse_dram_layout(graph.value("dram_layout", "plain")),
                   graph.value("logical_width", size_t{0}),
                   graph.value("logical_channels", size_t{0}),
                   graph.value("external_dram_layout", "compiler")};
    graph_infos[name] = info;
    if (!chained_graphs.count(name))
      runners_[name] = std::make_unique<Runner>(
          name, elf, info.inputs, info.outputs, info.output_layout,
          info.logical_width, info.logical_channels);
    if (verbose_)
      std::cout << "loaded " << name << " (" << graph.value("elf_bytes", 0) << " bytes)"
                << std::endl;
  }

  // Decode the steps once, so the request path does no JSON work.
  size_t step_index = 0;
  for (const auto& entry : plan.at("steps")) {
    Step step;
    step.opcode = parse_opcode(entry.at("op").get<std::string>());
    step.out = entry.at("out").get<std::string>();
    step.raw = entry.dump();
    const json args = entry.value("args", json::object());

    if (args.contains("graph")) step.graph = args.at("graph").get<std::string>();
    if (args.contains("graphs"))
      for (const auto& name : args.at("graphs")) step.graphs.push_back(name.get<std::string>());
    if (args.contains("src")) step.source = args.at("src").get<std::string>();
    if (args.contains("in"))
      for (const auto& name : args.at("in")) step.inputs.push_back(name.get<std::string>());
    if (args.contains("parts"))
      for (const auto& part : args.at("parts"))
        step.parts.push_back({part.at("src").get<std::string>(),
                              part.at("dst_offset").get<size_t>(),
                              part.at("count").get<size_t>(),
                              part.value("sidecar", false)});
    step.offset = args.value("offset", size_t{0});
    step.stride = args.value("stride", size_t{0});
    step.take = args.value("take", size_t{0});
    step.count = args.value("count", size_t{0});
    step.size = args.value("size", size_t{0});
    step.destination_stride = args.value("dst_stride", size_t{0});
    step.destination_offset = args.value("dst_offset", size_t{0});
    step.clear = args.value("clear", false);
    step.grid = args.value("grid", size_t{0});
    step.channels = args.value("channels", size_t{0});
    step.factor = args.value("factor", size_t{0});
    step.scalar = args.value("scalar", 0.0f);
    step.min_period = args.value("min_period", 0.0f);
    step.max_period = args.value("max_period", 0.0f);
    step.weights = args.value("weights", std::string{});
    step.bias = args.value("bias", std::string{});
    step.mean = args.value("mean", std::string{});
    step.std_dev = args.value("std", std::string{});
    step.rows = args.value("rows", size_t{0});
    step.cols = args.value("cols", size_t{0});

    if (!buffer_sizes_.count(step.out))
      throw std::runtime_error("step writes undeclared buffer '" + step.out + "'");
    if (step.opcode == Opcode::RunElf && !runners_.count(step.graph))
      throw std::runtime_error("step runs graph '" + step.graph + "' with no ELF in bundle.json");
    if (step.opcode == Opcode::RunElfChain) {
      if (step.graphs.size() < 2)
        throw std::runtime_error("run_elf_chain needs at least two graphs: " + step.raw);
      if (step.inputs.size() != 1)
        throw std::runtime_error("run_elf_chain needs exactly one input: " + step.raw);
      std::vector<SharedStage> stages;
      for (const auto& name : step.graphs) {
        auto found = graph_infos.find(name);
        if (found == graph_infos.end())
          throw std::runtime_error("shared chain graph '" + name + "' has no ELF");
        if (found->second.external_layout != "HWC")
          throw std::runtime_error("shared chain graph '" + name +
                                   "' was not compiled with external HWC layout");
        stages.push_back({name, found->second.elf, found->second.inputs, found->second.outputs});
      }
      const auto& tail = graph_infos.at(step.graphs.back());
      step.chain = "chain_" + std::to_string(step_index);
      chains_[step.chain] = std::make_unique<SharedRunnerChain>(
          step.chain, stages, tail.output_layout, tail.logical_width, tail.logical_channels);
      if (buffer_sizes_.at(step.inputs.front()) != chains_.at(step.chain)->input_elements() ||
          buffer_sizes_.at(step.out) != chains_.at(step.chain)->output_elements())
        throw std::runtime_error("run_elf_chain plan buffer sizes disagree with bundle metadata");
    }
    for (const auto& name : step.inputs)
      if (!buffer_sizes_.count(name))
        throw std::runtime_error("step reads undeclared buffer '" + name + "'");
    if (!step.source.empty() && !buffer_sizes_.count(step.source))
      throw std::runtime_error("step reads undeclared buffer '" + step.source + "'");
    for (const auto& part : step.parts)
      if (!part.sidecar && !buffer_sizes_.count(part.source))
        throw std::runtime_error("pack part reads undeclared buffer '" + part.source + "'");

    steps_.push_back(std::move(step));
    ++step_index;
  }
}

std::vector<float>& Plan::buffer(const std::string& name) {
  auto found = buffers_.find(name);
  if (found == buffers_.end()) throw std::runtime_error("no buffer '" + name + "'");
  return found->second;
}

const std::vector<float>& Plan::buffer(const std::string& name) const {
  auto found = buffers_.find(name);
  if (found == buffers_.end()) throw std::runtime_error("no buffer '" + name + "'");
  return found->second;
}

const std::vector<float>& Plan::read_buffer(const Step& step) const {
  if (!step.source.empty()) return buffer(step.source);
  if (!step.inputs.empty()) return buffer(step.inputs.front());
  throw std::runtime_error("step needs a source buffer but names none: " + step.raw);
}

void Plan::run_step(const Step& step) {
  auto& out = buffer(step.out);

  switch (step.opcode) {
    case Opcode::RunElf: {
      auto& runner = *runners_.at(step.graph);
      // A single input buffer is passed straight through; several are
      // concatenated (the plan validator guarantees the sizes line up).
      if (step.inputs.size() == 1) {
        runner.run(buffer(step.inputs[0]).data(), out.data());
      } else {
        std::vector<float> joined;
        joined.reserve(runner.input_elements());
        for (const auto& name : step.inputs) {
          const auto& source = buffer(name);
          joined.insert(joined.end(), source.begin(), source.end());
        }
        runner.run(joined.data(), out.data());
      }
      break;
    }

    case Opcode::RunElfChain: {
      chains_.at(step.chain)->run(buffer(step.inputs.front()).data(), out.data());
      break;
    }

    case Opcode::Pack: {
      // Zero first: the stem graph reads 601x512 but only ~307k of it is
      // written, and the latent token must be zero (EncoderStemLayer builds it
      // from torch.zeros).
      std::fill(out.begin(), out.end(), 0.0f);
      for (const auto& part : step.parts) {
        // A part comes either from a live buffer or from a constants/ sidecar.
        // SmolVLA's prefix needs both: two of its five sections are fixed for a
        // checkpoint (the empty-image and language embeddings), so they ship
        // with the bundle instead of crossing the wire every inference.
        const auto& source = part.sidecar ? sidecars_.get(part.source) : buffer(part.source);
        if (part.destination_offset + part.count > out.size())
          throw std::runtime_error("pack part overruns buffer '" + step.out + "'");
        if (part.count > source.size())
          throw std::runtime_error("pack part reads past source '" + part.source + "'");
        std::copy_n(source.begin(), part.count, out.begin() + part.destination_offset);
      }
      break;
    }

    case Opcode::Slice: {
      const auto& source = read_buffer(step);
      const size_t count = step.count ? step.count : out.size();
      if (step.offset + count > source.size())
        throw std::runtime_error("slice overruns its source on step " + step.raw);
      std::copy_n(source.begin() + step.offset, count, out.begin());
      break;
    }

    case Opcode::GatherStrided: {
      // ACT's 16 -> 6 unpad: DecoderActionRank4 widens the action head to 16
      // output channels for MLA channel alignment and zero-fills the rest, so
      // the host takes 6 of every 16 values, 100 times.
      //
      // SmolVLA needs the scatter direction too, so the destination stride is
      // configurable and defaults to `take` (which is ACT's contiguous case).
      // A source stride of 0 broadcasts one span into every block, which is how
      // the shared 720-wide time embedding reaches all 50 action tokens.
      const auto& source = read_buffer(step);
      const size_t destination_stride = step.destination_stride ? step.destination_stride
                                                                : step.take;
      if (step.clear) std::fill(out.begin(), out.end(), 0.0f);
      for (size_t index = 0; index < step.count; ++index) {
        const size_t read_at = index * step.stride;
        const size_t write_at = index * destination_stride + step.destination_offset;
        if (read_at + step.take > source.size() || write_at + step.take > out.size())
          throw std::runtime_error("gather_strided overruns on step " + step.raw);
        std::copy_n(source.begin() + read_at, step.take, out.begin() + write_at);
      }
      break;
    }

    case Opcode::PixelUnshuffle: {
      // Eagle's channel fold, between the vision chain and the connector: a
      // grid x grid map of C channels becomes (grid/f)^2 tokens of C*f^2,
      // matching torch.pixel_unshuffle's (c*f^2 + dy*f + dx) channel order.
      // The connector is the one Eagle graph the chain cannot swallow, because
      // this reshape has to happen between post_norm and it.
      const auto& source = read_buffer(step);
      const size_t grid = step.grid, channels = step.channels, factor = step.factor;
      if (!grid || !channels || !factor || grid % factor)
        throw std::runtime_error("pixel_unshuffle needs grid divisible by factor on step " +
                                 step.raw);
      const size_t side = grid / factor;
      const size_t out_channels = channels * factor * factor;
      if (source.size() != grid * grid * channels || out.size() != side * side * out_channels)
        throw std::runtime_error("pixel_unshuffle shape mismatch on step " + step.raw);
      for (size_t row = 0; row < side; ++row)
        for (size_t column = 0; column < side; ++column)
          for (size_t dy = 0; dy < factor; ++dy)
            for (size_t dx = 0; dx < factor; ++dx) {
              const size_t read_at =
                  ((row * factor + dy) * grid + (column * factor + dx)) * channels;
              const size_t write_at = (row * side + column) * out_channels + dy * factor + dx;
              for (size_t channel = 0; channel < channels; ++channel)
                out[write_at + channel * factor * factor] = source[read_at + channel];
            }
      break;
    }

    case Opcode::Scale: {
      const auto& source = read_buffer(step);
      for (size_t index = 0; index < out.size(); ++index) out[index] = source[index] * step.scalar;
      break;
    }

    case Opcode::MatVec: {
      const auto& x = read_buffer(step);
      const auto& w = sidecars_.get(step.weights);
      const size_t rows = step.rows ? step.rows : out.size();
      const size_t cols = step.cols ? step.cols : x.size();
      for (size_t row = 0; row < rows; ++row) {
        float total = step.bias.empty() ? 0.0f : sidecars_.get(step.bias)[row];
        const float* weights = w.data() + row * cols;
        for (size_t col = 0; col < cols; ++col) total += weights[col] * x[col];
        out[row] = total;
      }
      break;
    }

    case Opcode::SincosTime: {
      // Geometric period sweep from min_period to max_period, as an angular
      // frequency: period = min * (max/min)^(i/(half-1)), angle = t * 2pi/period.
      //
      // Transcribed from smolvla_som_server.cpp::make_suffix_input. The details
      // matter and are easy to get subtly wrong: the exponent divides by
      // half-1 (so the last entry lands exactly on max_period, not one step
      // short), and the angle carries the 2pi. Getting either wrong yields a
      // plausible-looking embedding and a policy that acts at the wrong phase
      // of the denoising schedule.
      const float t = step.scalar;
      const size_t half = out.size() / 2;
      if (half == 0) break;
      const float min_period = step.min_period > 0.0f ? step.min_period : 1.0f;
      const float max_period = step.max_period > 0.0f ? step.max_period : min_period;
      const float span = max_period / min_period;
      const float divisor = half > 1 ? static_cast<float>(half - 1) : 1.0f;
      for (size_t index = 0; index < half; ++index) {
        const float fraction = static_cast<float>(index) / divisor;
        const float period = min_period * std::pow(span, fraction);
        const float angle = t * (2.0f * static_cast<float>(M_PI) / period);
        out[index] = std::sin(angle);
        out[half + index] = std::cos(angle);
      }
      break;
    }

    case Opcode::Euler: {
      const auto& velocity = read_buffer(step);
      const float dt = step.scalar;
      for (size_t index = 0; index < out.size(); ++index) out[index] -= dt * velocity[index];
      break;
    }

    case Opcode::Normalize:
    case Opcode::Denormalize: {
      const auto& source = read_buffer(step);
      const auto& mean = sidecars_.get(step.mean);
      const auto& deviation = sidecars_.get(step.std_dev);
      for (size_t index = 0; index < out.size(); ++index) {
        const size_t stat = index % mean.size();
        out[index] = step.opcode == Opcode::Normalize
                         ? (source[index] - mean[stat]) / deviation[stat]
                         : source[index] * deviation[stat] + mean[stat];
      }
      break;
    }
  }
}

float* Plan::input_buffer(const std::string& name) { return buffer(name).data(); }

const std::vector<float>& Plan::execute() {
  timings_.clear();
  for (const auto& step : steps_) {
    const auto started = Clock::now();
    run_step(step);
    // Accelerator corruption or an incorrect DRAM layout must become a failed
    // request, never a motor command. ELF outputs are the useful diagnostic
    // boundaries; the final result also covers host-side arithmetic.
    if (step.opcode == Opcode::RunElf)
      require_finite("graph " + step.graph, buffer(step.out));
    if (step.opcode == Opcode::RunElfChain)
      require_finite("graph chain " + step.chain, buffer(step.out));
    if (collect_timings_)
      timings_.push_back(std::string(opcode_name(step.opcode)) + ":" +
                         (step.graph.empty() ? step.out : step.graph) + "=" +
                         std::to_string(elapsed_ms(started)));
  }
  require_finite("result " + result_, buffer(result_));
  return buffer(result_);
}

const std::vector<float>& Plan::execute(const std::map<std::string, const float*>& inputs) {
  for (const auto& [name, elements] : wire_.request_tensors) {
    auto supplied = inputs.find(name);
    if (supplied == inputs.end())
      throw std::runtime_error("missing input tensor '" + name + "'");
    auto& target = buffer(name);
    if (target.size() != elements)
      throw std::runtime_error("buffer '" + name + "' size disagrees with the wire description");
    std::memcpy(target.data(), supplied->second, elements * sizeof(float));
  }

  return execute();
}

}  // namespace polima
