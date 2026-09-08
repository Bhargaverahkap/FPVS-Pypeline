@echo off
REM ===================================================================
REM FPVS-Pypeline setup for Windows (run in cmd prompt)
REM Creates a virtual environment and installs all required libraries.
REM Usage:  setup_windows.bat
REM ===================================================================

setlocal

echo.
echo === Checking for Python ===
py -3 --version >nul 2>&1
if errorlevel 1 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo Python not found.
        echo Install Python 3.10+ from https://www.python.org/downloads/windows/
        echo Tick "Add python.exe to PATH" during installation, then run this again.
        exit /b 1
    )
    set "PY=python"
) else (
    set "PY=py -3"
)

%PY% --version

echo.
echo === Checking for Tk (needed by the desktop app) ===
%PY% -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo Python was installed without Tk/tcl support.
    echo Re-run the Python installer from https://www.python.org/downloads/windows/
    echo choose "Modify", and tick "tcl/tk and IDLE". Then run this script again.
    exit /b 1
)
echo Tk found.

echo.
echo === Creating virtual environment in .venv ===
if not exist ".venv" (
    %PY% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        exit /b 1
    )
) else (
    echo .venv already exists, reusing it.
)

echo.
echo === Installing libraries ===
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Library installation failed.
    exit /b 1
)

echo.
echo === Registering Jupyter kernel ===
python -m ipykernel install --user --name fpvs --display-name "Python (FPVS)"

echo.
echo === Done ===
echo Activate the environment in a new cmd window with:
echo     .venv\Scripts\activate.bat
echo Then start Jupyter with:
echo     jupyter lab
echo.

endlocal
