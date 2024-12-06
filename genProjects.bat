@echo off
if exist ".\build" rd /q /s ".\build" 2>nul
mkdir build
cd build
cmake ..