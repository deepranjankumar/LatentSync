import cv2
import numpy as np
import os
import ffmpeg

def extract_frames(video_path):
    """Extracts frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def write_video(output_path, frames, fps=30):
    """Writes a sequence of frames into a video file."""
    if len(frames) == 0:
        raise ValueError("No frames to write!")
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def add_audio_to_video(video_path, audio_source, output_path):
    """Adds audio from the input video to the generated video."""
    input_video = ffmpeg.input(video_path)
    input_audio = ffmpeg.input(audio_source).audio
    ffmpeg.output(input_video, input_audio, output_path, vcodec="copy", acodec="aac").run(overwrite_output=True)
