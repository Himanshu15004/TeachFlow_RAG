#Convert the videos to mp3

import os
import subprocess

files = os.listdir("videos")
for file in files:
    file_name = os.path.splitext(file)[0] 
    subprocess.run(["ffmpeg", "-i", f"Videos/{file}" , f"Audios/{file_name}.mp3"])