import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def read_image(image_path):
    """
    Read an image from a file path
    Args:
        image_path (str): Path to the image file
    Returns:
        List[np.array]: A list containing the image as a numpy array
    """
    return [cv2.imread(image_path)]


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

