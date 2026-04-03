@echo off
setlocal

cd /d "%~dp0"

set "PYTHONW=.venv\Scripts\pythonw.exe"
set "PYTHON=.venv\Scripts\python.exe"
set "SCRIPT=transcribe.py"

if exist "%PYTHONW%" (
    start "" "%PYTHONW%" "%SCRIPT%"
    exit /b 0
)

if exist "%PYTHON%" (
    start "" "%PYTHON%" "%SCRIPT%"
    exit /b 0
)

echo Виртуальное окружение не найдено: .venv\Scripts\python.exe
echo Создайте или восстановите .venv и установите зависимости перед запуском приложения.
pause
exit /b 1
