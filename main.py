import cv2
import torch
import imageio
import os

# Define paths and configurations
# Add the model in your system path (e.g., 'D:/custom/best_person.pt')
model_path = 'D:/custom/best_person.pt'
confidence_threshold = 0.9


def load_model(model_path):
    try:
        model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', model_path, force_reload=True)
        class_names = model.names
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def detect_and_save_video(path_of_video, save_folder, model, class_names):
    try:
        file_name, _ = os.path.splitext(os.path.basename(path_of_video))
        cap = cv2.VideoCapture(path_of_video)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {path_of_video}")

        save_path = os.path.join(save_folder, f"{file_name}_detected.mp4")
        writer = imageio.get_writer(save_path, fps=20)

        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % 3 != 0:
                continue
            frame = cv2.resize(frame, (1020, 600))

            results = model(frame)
            for det in results.pred[0]:
                class_id = int(det[5])
                class_name = class_names[class_id]
                confidence = det[4]
                bbox = det[:4]
                if class_name == "person" and confidence > confidence_threshold:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_name}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)

            cv2.imshow("FRAME", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        writer.close()
        cv2.destroyAllWindows()

        return save_path, count

    except Exception as e:
        print(f"Error during video processing: {e}")
        if cap:
            cap.release()
        if writer:
            writer.close()
        cv2.destroyAllWindows()
        raise

def FnPersonDetection(video_path):
    try:
        save_folder = os.path.dirname(video_path)
        model, class_names = load_model(model_path)
        result, detection_count = detect_and_save_video(video_path, save_folder, model, class_names)
        print(f"Output video saved at: {result}")
        print(f"Total person detections: {detection_count}")
        return result, detection_count
    except Exception as e:
        print(f"Error in person detection: {e}")
        raise

# Example usage
if __name__ == "__main__":
    video_path = 'path_to_your_video.mp4'
    FnPersonDetection(video_path)
