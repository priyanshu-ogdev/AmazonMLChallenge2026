@echo off
if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" goto has_ps
cmd.exe /c %*
goto :eof

:has_ps
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy Bypass %*
