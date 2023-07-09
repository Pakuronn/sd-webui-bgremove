import platform
import launch
import os

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            launch.run_pip(f"install {lib}", f"BgRemove requirement: {lib}")

# batch-face-swap requirements:
if not launch.is_installed("mediapipe") or not launch.is_installed("mediapipe-silicon"):
    name = "Batch Face Swap"
    if platform.system() == "Darwin" and platform.machine == "arm64":
        # MacOS
        launch.run_pip("install mediapipe-silicon", "requirements for Batch Face Swap for MacOS")
    else:
        launch.run_pip("install mediapipe", "requirements for Batch Face Swap")
