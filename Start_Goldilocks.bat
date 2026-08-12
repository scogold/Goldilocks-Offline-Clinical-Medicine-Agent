@echo off
setlocal
chcp 65001 >nul
title Goldilocks
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
    set "GOLDI_PYTHON=python"
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        set "GOLDI_PYTHON=py"
    ) else (
        echo No se encontró Python. Instálelo desde https://www.python.org/downloads/
        echo Python was not found. Install it from https://www.python.org/downloads/
        echo ^(marque "Add python.exe to PATH" durante la instalación / check "Add python.exe to PATH" during setup^)
        pause
        exit /b 1
    )
)

%GOLDI_PYTHON% launch.py
if errorlevel 1 (
    pause
)
