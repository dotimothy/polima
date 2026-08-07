// The robot side of polima-cli.
//
// The board runs the policy server and drives the arm; the host only compiles
// and deploys. So the robot commands belong in the binary already on the board,
// not in a second Python install shipped alongside it.
//
// What the board needs to know about the robot -- camera roles, which physical
// device fills each, joint order, fps -- travels in plan.json's `robot` section,
// written from RobotSpec at pack time. That is what lets this do discovery
// without a PolicySpec, and the board deliberately has no polima Python.
//
// The control loop itself is not here and cannot be: driving an SO-101 means
// lerobot, which is Python. polima-cli launches it in the board's
// /media/nvme/lerobot venv, the same way it launches polima_server.
#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace polima {

struct CameraDevice {
  std::string name;   // usb-046d_HD_Pro_Webcam_C920-video-index0
  std::string path;   // /dev/v4l/by-id/<name>
  std::string node;   // the /dev/videoN it resolves to
};

// The robot half of plan.json, written from RobotSpec at pack time.
struct RobotDescription {
  bool present = false;
  std::vector<std::pair<std::string, std::string>> roles;   // (role, label)
  std::map<std::string, std::string> hints;                 // role -> by-id substring
  std::string fourcc = "MJPG";
  std::string calibration_id;
  int fps = 30;
};

struct CameraAssignment {
  std::map<std::string, std::string> assigned;   // role -> device path
  std::vector<std::string> problems;
};

std::vector<CameraDevice> list_cameras(
    const std::filesystem::path& by_id = "/dev/v4l/by-id");
std::vector<std::string> list_serial_ports(const std::filesystem::path& dev = "/dev");

RobotDescription read_robot_description(const std::filesystem::path& bundle_root);
CameraAssignment assign_cameras(const RobotDescription& description,
                                const std::vector<CameraDevice>& cameras);

}  // namespace polima
