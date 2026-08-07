#include "polima/robot.hpp"

#include <algorithm>
#include <fstream>

#include <dirent.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace polima {
namespace fs = std::filesystem;
namespace {

using json = nlohmann::json;

bool ends_with(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string lowered(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  return text;
}

bool starts_with(const std::string& text, const std::string& prefix) {
  return text.rfind(prefix, 0) == 0;
}

}  // namespace

std::vector<CameraDevice> list_cameras(const fs::path& by_id) {
  std::vector<CameraDevice> cameras;
  std::error_code code;
  if (!fs::is_directory(by_id, code)) return cameras;

  for (const auto& entry : fs::directory_iterator(by_id, code)) {
    const std::string name = entry.path().filename().string();
    // Only the capture node. Higher indices are metadata streams: they open
    // fine and deliver nothing, which is a confusing way to get a black feed.
    if (!ends_with(name, "-video-index0")) continue;
    CameraDevice camera;
    camera.name = name;
    camera.path = entry.path().string();
    const fs::path resolved = fs::read_symlink(entry.path(), code);
    if (!code) camera.node = fs::weakly_canonical(by_id / resolved, code).string();
    cameras.push_back(std::move(camera));
  }
  std::sort(cameras.begin(), cameras.end(),
            [](const CameraDevice& a, const CameraDevice& b) { return a.name < b.name; });
  return cameras;
}

std::vector<std::string> list_serial_ports(const fs::path& dev) {
  std::vector<std::string> ports;
  std::error_code code;
  if (!fs::is_directory(dev, code)) return ports;
  for (const auto& entry : fs::directory_iterator(dev, code)) {
    const std::string name = entry.path().filename().string();
    if (starts_with(name, "ttyACM") || starts_with(name, "ttyUSB"))
      ports.push_back(entry.path().string());
  }
  std::sort(ports.begin(), ports.end());
  return ports;
}

RobotDescription read_robot_description(const fs::path& bundle_root) {
  RobotDescription description;
  std::ifstream stream(bundle_root / "plan.json");
  if (!stream) return description;
  json plan;
  try {
    stream >> plan;
  } catch (const std::exception&) {
    return description;
  }
  if (!plan.contains("robot")) return description;

  const json& robot = plan.at("robot");
  description.present = true;
  description.fps = robot.value("fps", 30);
  description.calibration_id = robot.value("calibration_id", "");
  description.fourcc = robot.value("camera_fourcc", "MJPG");
  for (const auto& pair : robot.value("camera_roles", json::array())) {
    if (pair.is_array() && pair.size() >= 2)
      description.roles.push_back({pair[0].get<std::string>(), pair[1].get<std::string>()});
  }
  // Bind the object once. `robot.value(...)` returns a fresh temporary on every
  // call, so iterating begin() to end() across two calls compares iterators
  // into two different containers.
  const json hints = robot.value("camera_hints", json::object());
  for (auto item = hints.begin(); item != hints.end(); ++item)
    description.hints[item.key()] = item.value().get<std::string>();
  return description;
}

CameraAssignment assign_cameras(const RobotDescription& description,
                                const std::vector<CameraDevice>& cameras) {
  CameraAssignment result;
  std::vector<std::string> taken;

  for (const auto& [role, label] : description.roles) {
    (void)label;
    const auto hint = description.hints.find(role);
    if (hint == description.hints.end() || hint->second.empty()) {
      result.problems.push_back(role + ": no camera hint in the bundle");
      continue;
    }
    std::vector<const CameraDevice*> matches;
    for (const auto& camera : cameras) {
      if (lowered(camera.name).find(lowered(hint->second)) == std::string::npos) continue;
      if (std::find(taken.begin(), taken.end(), camera.path) != taken.end()) continue;
      matches.push_back(&camera);
    }
    if (matches.empty()) {
      result.problems.push_back(role + ": no camera matching '" + hint->second + "'");
      continue;
    }
    if (matches.size() > 1) {
      result.problems.push_back(role + ": " + std::to_string(matches.size()) +
                                " cameras match '" + hint->second + "'");
      continue;
    }
    // Deliberately never falls back on enumeration order: /dev/videoN is
    // assigned in plug order, so a reboot can swap two cameras, and a swapped
    // pair raises nothing at all -- the policy runs and the arm reaches for the
    // wrong place.
    result.assigned[role] = matches.front()->path;
    taken.push_back(matches.front()->path);
  }
  return result;
}

}  // namespace polima
