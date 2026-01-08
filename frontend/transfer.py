import sys
import os

src = r"C:\Users\user\.gemini\antigravity\brain\cd103266-9cb3-48db-85f2-df09b6c6a2d3\bot_avatar_1767840584759.png"
dst = r"d:\rate\frontend\public\bot-avatar.png"

try:
    with open(src, "rb") as f_src:
        data = f_src.read()
    with open(dst, "wb") as f_dst:
        f_dst.write(data)
    print("SUCCESS")
except Exception as e:
    print(f"FAILURE: {e}")
