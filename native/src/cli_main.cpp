// polima_cli -- run a bundle once, files in, files out.
//
// The same plan interpreter as polima_server, without the socket. This is
// act_llima.cpp's --input-dir/--output branch, generalized: it is what the
// deploy smoke test and `polima run --local` use, and it is the fastest way to
// bisect a bad bundle because it can dump every intermediate buffer.
//
//   polima_cli --bundle /media/nvme/polima/current \
//              --input-dir <bundle>/fixtures/inputs \
//              --output /tmp/actions.f32 [--dump-stages /tmp/stages]

#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "polima/plan.hpp"
#include "polima/sidecar.hpp"
#include "polima/socket.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  try {
    polima::Args args(argc, argv);
    const fs::path bundle = args.get("--bundle");
    const bool verbose = args.has("--verbose");

    polima::Plan plan(bundle, verbose);
    const auto& wire = plan.wire();

    const fs::path input_dir = args.get("--input-dir", (bundle / "fixtures/inputs").string());
    const fs::path output = args.get("--output", "");

    // Read one .f32 per declared request tensor. Names match the wire
    // description, so fixtures/inputs/image0.f32 feeds tensor "image0".
    std::map<std::string, std::vector<float>> staging;
    std::map<std::string, const float*> inputs;
    for (const auto& [name, elements] : wire.request_tensors) {
      const fs::path path = input_dir / (name + ".f32");
      staging[name] = polima::read_f32(path, elements);
      inputs[name] = staging[name].data();
      if (verbose) std::cout << "read " << path << " (" << elements << " floats)" << std::endl;
    }

    const auto started = std::chrono::steady_clock::now();
    const auto& result = plan.execute(inputs);
    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();

    if (!output.empty()) {
      polima::write_f32(output, result);
      std::cout << "wrote " << output << " (" << result.size() << " floats)" << std::endl;
    }

    std::cout << "total_ms=" << total_ms << " count=" << result.size();
    for (const auto& timing : plan.stage_timings()) std::cout << " " << timing;
    std::cout << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << std::endl;
    return 1;
  }
}
