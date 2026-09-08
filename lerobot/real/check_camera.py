from pathlib import Path

from PIL import Image

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

from real_config import CAM_FPS, CAM_HEIGHT, CAM_WIDTH, CAMERA_INDEX

OUTPUT = Path(__file__).parent / "camera_check.png"


def main():
    config = OpenCVCameraConfig(
        index_or_path=CAMERA_INDEX,
        fps=CAM_FPS,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
    )
    camera = OpenCVCamera(config)
    camera.connect()
    image = camera.read()
    camera.disconnect()
    Image.fromarray(image).save(OUTPUT)
    print(f"Saved {OUTPUT} ({image.shape[1]}x{image.shape[0]})")
    print("Compare with the sim wrist view: gripper fingers at bottom center, mat ahead.")


if __name__ == "__main__":
    main()
