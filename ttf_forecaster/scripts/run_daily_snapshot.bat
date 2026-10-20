@echo off
rem Daily snapshot for Windows Task Scheduler: runs run_daily_snapshot.py and appends all output to
rem data\live\logs\daily_snapshot.log (git-ignored). Exit code 2 = Bloomberg Terminal not connected.
setlocal
set ROOT=%~dp0..
if not exist "%ROOT%\data\live\logs" mkdir "%ROOT%\data\live\logs"
set LOG=%ROOT%\data\live\logs\daily_snapshot.log
echo ===== %DATE% %TIME% start >> "%LOG%"
"%ROOT%\..\..\env_local\Scripts\python.exe" "%ROOT%\scripts\run_daily_snapshot.py" %* >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo ===== %DATE% %TIME% end, exit code %RC% >> "%LOG%"
exit /b %RC%
