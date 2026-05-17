Dim q : q = Chr(34)
Set sh = WScript.CreateObject("WScript.Shell")
sh.Run q & "C:\Program Files\PowerShell\7\pwsh.exe" & q & _
    " -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " & _
    q & "D:\test 2\AI trading assistance\scripts\zombie_watchdog.ps1" & q, 0, False
