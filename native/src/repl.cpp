#include "polima/repl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "polima/lineedit.hpp"
#include "polima/plan.hpp"
#include "polima/socket.hpp"

namespace polima {
namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;
namespace fs = std::filesystem;

double since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string human_bytes(size_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1);
  if (bytes >= (1u << 30)) out << bytes / 1073741824.0 << " GiB";
  else if (bytes >= (1u << 20)) out << bytes / 1048576.0 << " MiB";
  else out << bytes / 1024.0 << " KiB";
  return out.str();
}

// The MLA runtime prints two lines per ELF as it loads, so `use` on a six-graph
// bundle buries the one line that matters under eighteen. It is a library we do
// not control, so the only lever is the file descriptor. Errors still surface:
// a failed load throws, and the exception message is printed by the caller.
class Quiet {
 public:
  explicit Quiet(bool enable) {
    if (!enable) return;
    std::cout.flush();
    fflush(stdout);
    saved_out_ = dup(STDOUT_FILENO);
    saved_err_ = dup(STDERR_FILENO);
    const int null = open("/dev/null", O_WRONLY);
    if (null >= 0) {
      dup2(null, STDOUT_FILENO);
      dup2(null, STDERR_FILENO);
      close(null);
    }
  }
  ~Quiet() {
    if (saved_out_ < 0) return;
    fflush(stdout);
    fflush(stderr);
    dup2(saved_out_, STDOUT_FILENO);
    dup2(saved_err_, STDERR_FILENO);
    close(saved_out_);
    close(saved_err_);
  }
  Quiet(const Quiet&) = delete;
  Quiet& operator=(const Quiet&) = delete;

 private:
  int saved_out_ = -1;
  int saved_err_ = -1;
};

std::vector<std::string> split(const std::string& line) {
  std::istringstream stream(line);
  std::vector<std::string> parts;
  for (std::string word; stream >> word;) parts.push_back(word);
  return parts;
}

// Cosine and mean absolute error, the same pair the host-side smoke test uses.
struct Agreement {
  double cosine = 0.0;
  double mean_abs = 0.0;
  double max_abs = 0.0;
};

Agreement compare(const std::vector<float>& actual, const std::vector<float>& expected) {
  Agreement result;
  double dot = 0.0, left = 0.0, right = 0.0, total = 0.0;
  for (size_t index = 0; index < actual.size(); ++index) {
    dot += static_cast<double>(actual[index]) * expected[index];
    left += static_cast<double>(actual[index]) * actual[index];
    right += static_cast<double>(expected[index]) * expected[index];
    const double gap = std::fabs(actual[index] - expected[index]);
    total += gap;
    result.max_abs = std::max(result.max_abs, gap);
  }
  result.cosine = (left > 0 && right > 0) ? dot / (std::sqrt(left) * std::sqrt(right)) : 0.0;
  result.mean_abs = actual.empty() ? 0.0 : total / actual.size();
  return result;
}

void print_help() {
  std::cout <<
      "\n  models              list the board's model store\n"
      "  use <n|name>        load a model (the slow part; done once)\n"
      "  run                 one inference on the bundle's fixture inputs\n"
      "  bench [n=20]        time n inferences\n"
      "  stages              per-step timings from the last run\n"
      "  check               compare against the bundle's expected output\n"
      "  info                what is loaded: graphs, buffers, wire\n"
      "  save <file>         write the last result as float32\n"
      "  help, quit\n"
      "\n  arrows for history and cursor, tab to complete, Ctrl-C to abandon\n"
      "  a line or a running bench\n\n";
}

}  // namespace

std::vector<StoreEntry> scan_store(const fs::path& store) {
  std::vector<StoreEntry> entries;
  if (!fs::is_directory(store)) return entries;

  fs::path current;
  std::error_code code;
  const fs::path link = store.parent_path() / "current";
  if (fs::exists(link, code)) current = fs::canonical(link, code);

  for (const auto& item : fs::directory_iterator(store, code)) {
    if (!item.is_directory()) continue;
    StoreEntry entry;
    entry.name = item.path().filename().string();
    entry.root = item.path();
    entry.current = !current.empty() && fs::canonical(item.path(), code) == current;

    const fs::path manifest = item.path() / "bundle.json";
    if (fs::is_regular_file(manifest)) {
      entry.managed = true;
      try {
        std::ifstream stream(manifest);
        json value;
        stream >> value;
        entry.policy = value.value("policy", "");
        entry.graphs = value.contains("graphs") ? value.at("graphs").size() : 0;
        entry.elf_bytes = value.value("total_elf_bytes", size_t{0});
      } catch (const std::exception&) {
        entry.managed = false;   // present but unreadable; treat as unmanaged
      }
    }
    entries.push_back(std::move(entry));
  }

  // Loadable things first, then alphabetical, so the useful entries are at the
  // top of a store that also holds hand-built trees.
  std::sort(entries.begin(), entries.end(), [](const StoreEntry& a, const StoreEntry& b) {
    if (a.managed != b.managed) return a.managed;
    return a.name < b.name;
  });
  return entries;
}

int repl(const fs::path& store, const std::string& preselect, bool verbose) {
  auto entries = scan_store(store);

  auto show_models = [&entries, &store]() {
    if (entries.empty()) {
      std::cout << "  nothing in " << store.string() << "\n";
      return;
    }
    std::cout << "\n  " << std::left << std::setw(4) << "#" << std::setw(32) << "name"
              << std::setw(10) << "policy" << std::setw(9) << "graphs"
              << std::setw(12) << "elf" << "\n";
    for (size_t index = 0; index < entries.size(); ++index) {
      const auto& entry = entries[index];
      std::cout << "  " << std::left << std::setw(4) << (index + 1)
                << std::setw(32) << entry.name
                << std::setw(10) << (entry.policy.empty() ? "-" : entry.policy)
                << std::setw(9) << (entry.managed ? std::to_string(entry.graphs) : "-")
                << std::setw(12) << (entry.elf_bytes ? human_bytes(entry.elf_bytes) : "-");
      if (!entry.managed) std::cout << " legacy tree";
      if (entry.current) std::cout << " <- current";
      std::cout << "\n";
    }
    std::cout << "\n";
  };

  install_interrupt_handler();
  LineEditor editor;
  const std::vector<std::string> commands = {
      "models", "use", "run", "bench", "stages", "check", "info", "save", "help", "quit"};
  auto refresh_completions = [&]() {
    std::vector<std::string> words = commands;
    for (const auto& entry : entries)
      if (entry.managed) words.push_back(entry.name);
    editor.set_completions(std::move(words));
  };

  std::unique_ptr<Plan> plan;
  std::string loaded;
  std::vector<float> last_result;
  std::map<std::string, std::vector<float>> staging;

  auto load = [&](const std::string& token) -> bool {
    const StoreEntry* chosen = nullptr;
    // Accept an index or a name, because typing a content-addressed id is not
    // something anyone should have to do.
    if (!token.empty() && std::all_of(token.begin(), token.end(), ::isdigit)) {
      const size_t index = std::stoul(token);
      if (index >= 1 && index <= entries.size()) chosen = &entries[index - 1];
    } else {
      for (const auto& entry : entries)
        if (entry.name == token) chosen = &entry;
    }
    if (chosen == nullptr) {
      std::cout << "  no such model: " << token << "  (try `models`)\n";
      return false;
    }
    if (!chosen->managed) {
      std::cout << "  " << chosen->name
                << " is a hand-built tree with no plan.json, so polima-cli cannot"
                   " run it.\n  Serve it with its own launcher, or rebuild it with"
                   " `polima compile`.\n";
      return false;
    }
    try {
      const auto started = Clock::now();
      std::unique_ptr<Plan> candidate;
      {
        Quiet quiet(!verbose);
        candidate = std::make_unique<Plan>(chosen->root, verbose);
      }
      const double ms = since(started);
      plan = std::move(candidate);
      loaded = chosen->name;
      last_result.clear();
      staging.clear();
      std::cout << "  loaded " << loaded << " (" << chosen->graphs << " graphs, "
                << std::fixed << std::setprecision(1) << ms / 1000.0 << "s)\n";
      return true;
    } catch (const std::exception& error) {
      std::cout << "  failed to load " << chosen->name << ": " << error.what() << "\n";
      return false;
    }
  };

  // Read the fixture inputs once per loaded model; every run reuses them.
  auto ensure_inputs = [&]() -> bool {
    if (!staging.empty()) return true;
    const fs::path directory = fs::path(store) / loaded / "fixtures" / "inputs";
    try {
      for (const auto& [name, elements] : plan->wire().request_tensors)
        staging[name] = read_f32(directory / (name + ".f32"), elements);
      return true;
    } catch (const std::exception& error) {
      staging.clear();
      std::cout << "  no usable fixture inputs in " << directory.string() << ": " << error.what() << "\n";
      return false;
    }
  };

  auto execute = [&]() -> double {
    std::map<std::string, const float*> inputs;
    for (auto& [name, values] : staging) inputs[name] = values.data();
    const auto started = Clock::now();
    last_result = plan->execute(inputs);
    return since(started);
  };

  std::cout << "PoLiMa interactive  --  " << store.string() << "\n";
  show_models();
  if (!preselect.empty()) load(preselect);
  else if (entries.size() == 1 && entries[0].managed) load(entries[0].name);
  std::cout << "type `help` for commands\n";

  refresh_completions();
  std::string line;
  while (true) {
    g_interrupted = 0;
    if (!editor.read((loaded.empty() ? "polima" : loaded) + "> ", line)) break;
    const auto words = split(line);
    if (words.empty()) continue;
    const std::string& command = words[0];

    try {
      if (command == "quit" || command == "exit" || command == "q") break;
      if (command == "help" || command == "?") { print_help(); continue; }
      if (command == "models" || command == "ls") {
        entries = scan_store(store);
        refresh_completions();
        show_models();
        continue;
      }
      if (command == "use") {
        if (words.size() < 2) {
          show_models();
          std::cout << "  usage: use <n|name>\n";
          continue;
        }
        if (load(words[1])) refresh_completions();
        continue;
      }

      if (plan == nullptr) {
        std::cout << "  no model loaded -- `use <n>` first (`models` to list)\n";
        continue;
      }

      if (command == "info") {
        std::cout << "  bundle    " << plan->bundle_id() << "\n"
                  << "  policy    " << plan->policy() << "\n"
                  << "  result    " << plan->result_elements() << " floats\n"
                  << "  wire      port " << plan->wire().default_port << ", "
                  << plan->wire().request_tensors.size() << " input tensor(s)\n";
        for (const auto& [name, elements] : plan->wire().request_tensors)
          std::cout << "              " << name << "  " << elements << " floats\n";
        continue;
      }

      if (command == "run") {
        if (!ensure_inputs()) continue;
        const double ms = execute();
        std::cout << "  " << last_result.size() << " floats in "
                  << std::fixed << std::setprecision(1) << ms << " ms\n";
        continue;
      }

      if (command == "bench") {
        if (!ensure_inputs()) continue;
        const int count = words.size() > 1 ? std::max(1, std::stoi(words[1])) : 20;
        std::vector<double> samples;
        samples.reserve(count);
        for (int index = 0; index < count && !g_interrupted; ++index)
          samples.push_back(execute());
        if (g_interrupted) {
          std::cout << "  interrupted after " << samples.size() << " run(s)\n";
          g_interrupted = 0;
          if (samples.empty()) continue;
        }
        std::sort(samples.begin(), samples.end());
        const double mean =
            std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        std::cout << std::fixed << std::setprecision(1)
                  << "  min " << samples.front() << " / mean " << mean
                  << " / median " << samples[samples.size() / 2]
                  << " / max " << samples.back() << " ms over " << samples.size()
                  << " runs\n";
        continue;
      }

      if (command == "stages") {
        // Run once with timing on rather than asking for a restart with
        // --verbose. Timing costs a string per step, so it goes straight back
        // off: on SmolVLA that is 82 allocations per inference.
        if (!ensure_inputs()) continue;
        plan->set_collect_timings(true);
        const double ms = execute();
        plan->set_collect_timings(verbose);
        std::cout << std::fixed << std::setprecision(1) << "  total " << ms << " ms\n";
        for (const auto& timing : plan->stage_timings())
          std::cout << "    " << timing << "\n";
        continue;
      }

      if (command == "check") {
        if (last_result.empty()) { std::cout << "  nothing to check -- `run` first\n"; continue; }
        const fs::path expected =
            fs::path(store) / loaded / "fixtures" / "expected" / "normalized_actions.f32";
        if (!fs::is_regular_file(expected)) {
          std::cout << "  no reference at " << expected.string() << "\n";
          continue;
        }
        const auto reference = read_f32(expected, last_result.size());
        const auto agreement = compare(last_result, reference);
        const bool ok = agreement.cosine >= 0.999 && agreement.mean_abs <= 0.01;
        std::cout << (ok ? "  PASS  " : "  FAIL  ")
                  << std::fixed << std::setprecision(6)
                  << "cosine=" << agreement.cosine
                  << std::setprecision(4)
                  << " mean_abs=" << agreement.mean_abs
                  << " max_abs=" << agreement.max_abs << "\n";
        continue;
      }

      if (command == "save") {
        if (last_result.empty()) { std::cout << "  nothing to save -- `run` first\n"; continue; }
        if (words.size() < 2) { std::cout << "  usage: save <file>\n"; continue; }
        write_f32(words[1], last_result);
        std::cout << "  wrote " << words[1] << " (" << last_result.size() << " floats)\n";
        continue;
      }

      std::cout << "  unknown command: " << command << "  (try `help`)\n";
    } catch (const std::exception& error) {
      // A bad command must not end the session -- the whole point is that the
      // model stays loaded.
      std::cout << "  error: " << error.what() << "\n";
    }
  }
  return 0;
}

}  // namespace polima
