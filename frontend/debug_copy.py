import shutil
import os

src = r"C:\Users\user\.gemini\antigravity\brain\cd103266-9cb3-48db-85f2-df09b6c6a2d3\bot_avatar_1767840584759.png"
dst = r"d:\rate\frontend\public\bot-avatar.png"
log_file = r"d:\rate\frontend\copy_log.txt"

with open(log_file, "w") as f:
    f.write(f"Starting copy...\n")
    f.write(f"CWD: {os.getcwd()}\n")
    f.write(f"Source: {src}\n")
    if os.path.exists(src):
        f.write("Source exists.\n")
    else:
        f.write("Source DOES NOT exist.\n")

    try:
        shutil.copy(src, dst)
        f.write(f"Copy command executed to {dst}\n")
        if os.path.exists(dst):
             f.write("Destination file exists.\n")
             f.write(f"Size: {os.path.getsize(dst)}\n")
        else:
             f.write("Destination file NOT found after copy.\n")
    except Exception as e:
        f.write(f"Error during copy: {e}\n")
