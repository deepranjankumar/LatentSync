This project uses LatentSync, GFPGAN, and CodeFormer to perform lip-syncing and super-resolution on videos.

📂 Project Structure
  Superres/
│── LatentSync/               # Cloned LatentSync repository
│── CodeFormer/               # Cloned CodeFormer repository
│── experiments/
│   ├── pretrained_models/    # Pre-trained models go here
│── utils/
│   ├── lipsync.py            # Handles lip-sync processing
│   ├── superres.py           # Functions for super-resolution
│   ├── video_utils.py        # Video processing utilities
│── env/                      # Virtual environment
│── script.py                 # Main script to process video
│── requirements.txt          # Required dependencies
│── ReadMe.md                 # Instructions (this file)


Installation & Setup

1️⃣ Clone Required Repositories
git clone https://github.com/bytedance/LatentSync.git
cd LatentSync
pip install -e .  # Install LatentSync dependencies
cd ..

git clone https://github.com/sczhou/CodeFormer.git
cd CodeFormer
pip install -e .  # Install CodeFormer dependencies
cd ..

2️⃣ Create and Activate a Virtual Environment
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows

3️⃣ Install Required Packages
pip install -r requirements.txt

4️⃣ Download Pretrained Models
Ensure these files are placed in experiments/pretrained_models/:

GFPGANv1.4.pth (For GFPGAN super-resolution)

codeformer.pth (For CodeFormer super-resolution)

5️⃣ Run the Script
python script.py --input input_with_audio.mp4 --output output_video.mp4 --superres GFPGAN

📌 Notes

Tested on Python 3.8+

Requires a GPU for faster processing