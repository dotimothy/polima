// The interactive half of polima-cli.
//
// `llima run <model>` loads a model once and hands you a session; the expensive
// part (weights onto the MLA) happens once and every prompt after that is fast.
// polima-cli was one-shot, which meant a ~1s load for each 20ms inference and no
// way to poke at a bundle without composing a command line.
//
// So: no arguments, or `--interactive`, opens a session over the board's model
// store. `--bundle` still runs once and exits, because scripts and the deploy
// smoke test depend on that.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace polima {

// One entry in the board's model store.
struct StoreEntry {
  std::string name;
  std::filesystem::path root;
  std::string policy;        // empty for a hand-built tree with no bundle.json
  size_t graphs = 0;
  size_t elf_bytes = 0;
  bool managed = false;      // has bundle.json -> PoLiMa built it
  bool current = false;      // the `current` symlink points here
};

// Everything in `store` that looks like a servable model tree, sorted with
// PoLiMa bundles first. Legacy trees are listed but cannot be loaded, because
// they carry no plan.json -- the listing says so rather than failing later.
std::vector<StoreEntry> scan_store(const std::filesystem::path& store);

// Run the interactive session. Returns a process exit code.
int repl(const std::filesystem::path& store, const std::string& preselect, bool verbose);

}  // namespace polima
