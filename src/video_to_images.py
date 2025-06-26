import cv2
import os

def video_to_images(video_path,output_folder):
    # Opening the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video name (without extension)
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save each frame as PNG
        img_name = f"{vid_name}_img{frame_count:04d}.png"
        img_path = os.path.join(output_folder, img_name)
        cv2.imwrite(img_path, frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'")

if __name__ == "__main__":
    video_path = r"D:\Downloads\New York City (NYC) Night Traffic Video _ Free Content _ Open Source _ No Copyright _ Video Material.mp4"  # ✅ Fixed path
    output_folder = "output_frames"
    video_to_images(video_path, output_folder)