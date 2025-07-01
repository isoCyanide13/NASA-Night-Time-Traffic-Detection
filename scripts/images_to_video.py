import cv2
import os

def images_to_video(image_folder, output_video, fps=30):
    # Get all PNG files in the folder and sort them
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure correct order (0000, 0001, ...)
    
    if not images:
        print("Error: No PNG images found in the folder.")
        return
    
    # Read first image to get dimensions
    first_img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_img.shape
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print(f"Video saved as '{output_video}'")

if __name__ == "__main__":
    image_folder = "output_frames"  # Folder containing PNG images
    output_video = "output_video2.mp4"  # Output video name
    fps = 24  # Frames per second (adjust as needed)
    images_to_video(image_folder, output_video, fps)