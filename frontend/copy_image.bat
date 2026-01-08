@echo off
copy /Y "C:\Users\user\.gemini\antigravity\brain\cd103266-9cb3-48db-85f2-df09b6c6a2d3\bot_avatar_1767840584759.png" "d:\rate\frontend\public\bot-avatar.png" > copy_output.txt 2>&1
if %errorlevel% neq 0 (
    echo "Failed" >> copy_output.txt
) else (
    echo "Success" >> copy_output.txt
)
