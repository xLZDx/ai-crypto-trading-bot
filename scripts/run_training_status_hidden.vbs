Dim q : q = Chr(34)
Set sh = WScript.CreateObject("WScript.Shell")
sh.Run "cmd.exe /c " & q & _
    "D:\test 2\AI trading assistance\scripts\scheduled\AI-Trader-TrainingStatus.cmd" & q, 0, False
