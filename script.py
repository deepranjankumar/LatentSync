import argparse
import cv2
import os
import importlib.util

from utils.lipsync import process_lipsync, get_lipsynced_region
from utils.superres import load_sr_model, apply_super_resolution
from utils.video_utils import extract_frames, write_video, add_audio_to_video

# Load CodeFormer dynamically
spec = importlib.util.spec_from_file_location(
    "CodeFormer.basicsr.archs.codeformer_arch",
    r"C:\Superres\CodeFormer\basicsr\archs\codeformer_arch.py"
)
codeformer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codeformer_module)
CodeFormer = codeformer_module.CodeFormer

def main(args):
    sr_model = load_sr_model(args.superres)

    original_frames = extract_frames(args.input)
    lipsynced_frames = extract_frames(args.lipsynced)

    output_frames = []
    for orig_frame, lip_frame in zip(original_frames, lipsynced_frames):
        # Process lipsync
        processed_frame = process_lipsync(orig_frame, lip_frame)

        # Get lipsynced region (ROI)
        roi, bbox = get_lipsynced_region(orig_frame, processed_frame)

        # Check if super-resolution is needed
        orig_h, orig_w = orig_frame.shape[:2]
        roi_h, roi_w = roi.shape[:2]

        if roi_w < orig_w and roi_h < orig_h:
            print(f"Applying {args.superres} super-resolution on generated region...")
            roi = apply_super_resolution(sr_model, roi)

            # Place enhanced ROI back into frame
            x, y, w, h = bbox
            processed_frame[y:y+h, x:x+w] = roi

        output_frames.append(processed_frame)

    # Save output video
    temp_output = "temp_output.mp4"
    write_video(temp_output, output_frames)

    # Merge audio back
    add_audio_to_video(temp_output, args.input, args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input video with audio')
    parser.add_argument('--lipsynced', type=str, required=True, help='Path to lipsynced video (without audio)')
    parser.add_argument('--output', type=str, required=True, help='Path to final output video')
    parser.add_argument('--superres', type=str, choices=['GFPGAN', 'CodeFormer', 'None'], default='None', help='Super-resolution method')
    args = parser.parse_args()

    main(args)
