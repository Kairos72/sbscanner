@echo off
REM Windows batch file to run the Major Pairs real-time monitor
REM This will keep the service running and restart if it crashes

TITLE Major Pairs Real-time Monitor

:LOOP
echo Starting Major Pairs Real-time Monitor...
echo [%date% %time%] Service started >> major_pairs_monitor_service.log

REM Run the Python script
python realtime_major_pairs_service.py

REM Check exit code
IF %ERRORLEVEL% EQU 0 (
    echo Service shutdown normally
    echo [%date% %time%] Service shutdown normally >> major_pairs_monitor_service.log
) ELSE (
    echo Service crashed with error code %ERRORLEVEL%
    echo [%date% %time%] Service crashed with error code %ERRORLEVEL% >> major_pairs_monitor_service.log
    echo Restarting in 10 seconds...
    timeout /t 10 /nobreak > nul
    goto LOOP
)

pause