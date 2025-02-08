import argparse
import cv2
import sys
import os

from utils.lipsync import process_lipsync
from utils.superres import load_sr_model, apply_super_resolution
from utils.video_utils import extract_frames, write_video, add_audio_to_video

import sys
import os

import importlib.util

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
        processed_frame = process_lipsync(orig_frame, lip_frame)

        if sr_model:
            processed_frame = apply_super_resolution(sr_model, processed_frame)

        output_frames.append(processed_frame)

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
