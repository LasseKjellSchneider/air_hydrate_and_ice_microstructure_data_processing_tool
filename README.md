# Beyond Epica Hydrate Analysis

Desktop application for hydrate detection, grain-boundary segmentation, calibration, layer alignment, fabric analysis, and 3D reconstruction.

## Requirements

- Windows
- Python
- Packages listed in requirements.txt

## Setup and launch

From this directory, run the following in PowerShell:

`powershell`
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src\gui_main.py


The grain-boundary processing code uses CuPy when a compatible CUDA GPU is available and falls back to CPU processing otherwise. The dependency list includes the CUDA 12 CuPy package; installation and GPU acceleration depend on the local Python, CUDA, and NVIDIA driver environment.

## Project layout

- `src\gui_main.py`: application entry point
- `src\gui_modules\`: desktop workflow screens
- `src\core_backend\`: image processing, analysis, and reconstruction logic
- `requirements.txt`: Python dependencies
- `hydrate_constants.json` contains the hydrate-processing and calibration settings.
- `src\grain_boundary_constants.json` contains grain-boundary processing defaults. The application may update this file when parameters are saved.
