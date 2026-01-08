$src = "C:\Users\user\.gemini\antigravity\brain\cd103266-9cb3-48db-85f2-df09b6c6a2d3\bot_avatar_1767840584759.png"
$dest = "d:\rate\frontend\public\bot-avatar.png"
try {
    [System.IO.File]::Copy($src, $dest, $true)
    "Success" | Out-File "d:\rate\frontend\ps_output.txt"
} catch {
    $_.Exception.Message | Out-File "d:\rate\frontend\ps_output.txt"
}
