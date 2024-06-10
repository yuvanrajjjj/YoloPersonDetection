# YoloPersonDetection
# Person Detection in Videos

This project utilizes YOLOv5 to detect persons in video files. It processes each frame of the video, draws bounding boxes around detected persons, and saves the result as a new video file.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yuvanrajjjj/YoloPersonDetection.git
    cd YoloPersonDetection
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To detect persons in a video, run the script with the path to the video file:

```bash
python main.py path_to_your_video.mp4
