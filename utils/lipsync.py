import cv2

def process_lipsync(original_frame, lipsynced_frame):
    """
    Overlays the lipsynced region from `lipsynced_frame` onto `original_frame`.
    """
    h, w, _ = original_frame.shape
    lipsynced_frame = cv2.resize(lipsynced_frame, (w, h))

    # Replace the face area (assuming lipsync only affects the face region)
    blended_frame = original_frame.copy()
    blended_frame[:h, :w] = lipsynced_frame  # Modify this to blend properly if needed

    return blended_frame
