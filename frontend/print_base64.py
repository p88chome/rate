import base64
try:
    with open(r"C:\Users\user\.gemini\antigravity\brain\cd103266-9cb3-48db-85f2-df09b6c6a2d3\bot_avatar_1767840584759.png", "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
        print(encoded)
except Exception as e:
    print(f"Error: {e}")
