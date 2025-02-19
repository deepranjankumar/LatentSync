import argparse
import cv2
import os
import importlib.util
import tempfile

from utils.lipsync import process_lipsync, get_lipsynced_region
from utils.superres import apply_super_resolution
from utils.video_utils import extract_frames, write_video, add_audio_to_video

def load_sr_model(model_name):
    """Loads the chosen super-resolution model (GFPGAN or CodeFormer)."""
    if model_name == "GFPGAN":
        try:
            from gfpgan import GFPGANer
            model = GFPGANer(model_path="path/to/gfpgan.pth", upscale=2)
            print("Loaded GFPGAN successfully.")
            return model
        except ImportError:
            print("Error: GFPGAN not installed. Run `pip install gfpgan`.")
            return None
    elif model_name == "CodeFormer":
        try:
            # Dynamically load CodeFormer
            spec = importlib.util.spec_from_file_location(
                "CodeFormer.basicsr.archs.codeformer_arch",
                r"C:\Superres\CodeFormer\basicsr\archs\codeformer_arch.py"
            )
            codeformer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(codeformer_module)
            CodeFormer = codeformer_module.CodeFormer
            print("Loaded CodeFormer successfully.")
            return CodeFormer()
        except Exception as e:
            print(f"Error loading CodeFormer: {e}")
            return None
    else:
        print("No super-resolution selected.")
        return None

def main(args):
    # Load super-resolution model
    sr_model = load_sr_model(args.superres)

    # Extract frames from both videos
    original_frames = extract_frames(args.input)
    lipsynced_frames = extract_frames(args.lipsynced)

    output_frames = []
    for orig_frame, lip_frame in zip(original_frames, lipsynced_frames):
        # Apply lipsync
        processed_frame = process_lipsync(orig_frame, lip_frame)

        # Get lipsynced region (ROI)
        roi, bbox = get_lipsynced_region(orig_frame, processed_frame)

        # Apply super-resolution if needed
        if sr_model and roi is not None:
            print(f"Applying {args.superres} super-resolution on generated region...")
            roi = apply_super_resolution(sr_model, roi, method=args.superres)

            # Place enhanced ROI back into the frame
            x, y, w, h = bbox
            processed_frame[y:y+h, x:x+w] = roi

        output_frames.append(processed_frame)

    # Create a temporary output video
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    write_video(temp_output, output_frames)

    # Merge audio back
    add_audio_to_video(temp_output, args.input, args.output)

    # Cleanup temporary files
    os.remove(temp_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input video with audio')
    parser.add_argument('--lipsynced', type=str, required=True, help='Path to lipsynced video (without audio)')
    parser.add_argument('--output', type=str, required=True, help='Path to final output video')
    parser.add_argument('--superres', type=str, choices=['GFPGAN', 'CodeFormer', 'None'], default='None', help='Super-resolution method')
    
    args = parser.parse_args()
    main(args)
