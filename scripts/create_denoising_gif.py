#!/usr/bin/env python3
"""
Create a video from denoising process images with step numbers overlaid.
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import glob
import cv2
import numpy as np

def create_denoising_video(input_dir, output_path, fps=10):
    """
    Create a video from denoising images with step numbers.
    
    Args:
        input_dir: Directory containing the denoising images
        output_path: Output path for the video (MP4 or AVI)
        fps: Frames per second (default: 10)
    """
    # Get all images sorted by step number
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Try to load a decent font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    # Get dimensions from first image
    first_img = Image.open(image_files[0])
    height, width = first_img.size[1], first_img.size[0]
    
    # Determine codec and extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        # Default to MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
    
    # Create video writer
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for idx, img_path in enumerate(image_files):
        # Extract step number from filename (e.g., 0000_*.png -> 0)
        filename = os.path.basename(img_path)
        step_num = int(filename.split('_')[0])
        
        # Open image and convert to RGB (for GIF compatibility)
        img = Image.open(img_path).convert('RGB')
        
        # Create a copy to draw on
        img_with_text = img.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # Add step number text at the top
        text = f"{step_num}"
        
        # Get text bounding box for background rectangle
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position at top center with some padding
        x = (img.width - text_width) // 2
        y = 20
        
        # Draw semi-transparent background rectangle
        padding = 10
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 200)
        )
        
        # Draw text in white
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Convert PIL image to OpenCV format (RGB to BGR)
        frame_cv = cv2.cvtColor(np.array(img_with_text), cv2.COLOR_RGB2BGR)
        video_writer.write(frame_cv)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images")
    
    # Release video writer
    video_writer.release()
    
    print(f"\n✓ Video created successfully!")
    print(f"  Total frames: {len(image_files)}")
    print(f"  FPS: {fps}")
    print(f"  Total duration: {len(image_files) / fps:.1f}s")
    print(f"  Output: {output_path}")

if __name__ == "__main__":
    input_dir = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-12/02-35-03/trajectories"
    output_path = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-12/02-35-03/trajectories/denoising_process1_mi_no_floor.mp4"
    
    # 10 fps - good balance between smooth and not too fast
    create_denoising_video(input_dir, output_path, fps=10)
