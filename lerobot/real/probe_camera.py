from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

from real_config import CAMERA_INDEX

CANDIDATES = [
    {"width": 640, "height": 480, "fps": 30},
    {"width": 640, "height": 480, "fps": 15},
    {"width": 640, "height": 480},
    {"width": 1280, "height": 720, "fps": 30},
    {"width": 1920, "height": 1080, "fps": 30},
    {},
]


def main():
    for candidate in CANDIDATES:
        try:
            config = OpenCVCameraConfig(index_or_path=CAMERA_INDEX, **candidate)
            camera = OpenCVCamera(config)
            camera.connect()
            image = camera.read()
            camera.disconnect()
            print(f"OK   {candidate} -> frame {image.shape[1]}x{image.shape[0]}")
        except Exception as e:
            print(f"FAIL {candidate} -> {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
