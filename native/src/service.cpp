#include "polima/service.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace polima {
namespace fs = std::filesystem;
namespace {

fs::path pid_path(const fs::path& root) { return root / "var" / "run" / "server.pid"; }
fs::path log_path(const fs::path& root) { return root / "var" / "log" / "server.log"; }

bool alive(int pid) { return pid > 0 && ::kill(pid, 0) == 0; }

}  // namespace

ServerState server_state(const fs::path& root) {
  ServerState state;
  std::ifstream stream(pid_path(root));
  if (stream) stream >> state.pid;
  state.running = alive(state.pid);
  if (!state.running) state.pid = 0;

  std::error_code code;
  const fs::path current = root / "current";
  if (fs::exists(current, code)) {
    const fs::path resolved = fs::canonical(current, code);
    if (!code) state.bundle = resolved.filename().string();
  }
  return state;
}

bool stop_server(const fs::path& root, double timeout_s) {
  const ServerState state = server_state(root);
  if (!state.running) {
    std::error_code code;
    fs::remove(pid_path(root), code);
    return false;
  }

  ::kill(state.pid, SIGTERM);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int>(timeout_s * 1000));
  while (std::chrono::steady_clock::now() < deadline) {
    if (!alive(state.pid)) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  // A server mid-inference will not see SIGTERM until the MLA call returns, so
  // the wait above is not optional -- but it cannot be unbounded either.
  if (alive(state.pid)) ::kill(state.pid, SIGKILL);

  std::error_code code;
  fs::remove(pid_path(root), code);
  return true;
}

int start_server(const fs::path& root, const fs::path& bundle, int port) {
  const fs::path binary = root / "bin" / "polima_server";
  if (!fs::exists(binary)) return 0;

  std::error_code code;
  fs::create_directories(log_path(root).parent_path(), code);
  fs::create_directories(pid_path(root).parent_path(), code);

  const pid_t pid = ::fork();
  if (pid < 0) return 0;
  if (pid == 0) {
    // Detach fully: new session, no controlling terminal, stdin closed. The
    // shell version of this needed `< /dev/null` or the server inherited the
    // ssh channel and held the connection open; the same applies here.
    ::setsid();
    const int null = ::open("/dev/null", O_RDONLY);
    if (null >= 0) { ::dup2(null, STDIN_FILENO); ::close(null); }
    const int log = ::open(log_path(root).c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (log >= 0) {
      ::dup2(log, STDOUT_FILENO);
      ::dup2(log, STDERR_FILENO);
      ::close(log);
    }
    const std::string port_text = std::to_string(port);
    ::execl(binary.c_str(), "polima_server", "--bundle", bundle.c_str(),
            "--port", port_text.c_str(), static_cast<char*>(nullptr));
    ::_exit(127);
  }

  std::ofstream stream(pid_path(root));
  stream << pid << "\n";
  return pid;
}

}  // namespace polima
