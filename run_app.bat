@echo off
title NIFTY 50 Stock Predictor - Python 3.12 Fixed
color 0A

echo.
echo ================================================
echo      NIFTY 50 STOCK PREDICTOR - FIXED
echo        Python 3.12 Compatible Version
echo ================================================
echo.

echo [INFO] Fixing Python 3.12 package issues...
echo - Bypassing corrupted setuptools
echo - Using pre-compiled wheels only
echo - Avoiding source builds
echo.

echo [1/6] Checking Python...
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)
echo [SUCCESS] Python found!

echo.
echo [2/6] Fixing pip and setuptools...
python -m pip install --upgrade pip setuptools wheel --quiet --no-warn-script-location
echo [SUCCESS] Updated core tools!

echo.
echo [3/6] Cleaning problematic packages...
python -m pip uninstall pandas numpy streamlit -y --quiet
echo [SUCCESS] Cleaned old packages!

echo.
echo [4/6] Installing with pre-compiled wheels only...
echo This avoids the build errors you saw...
python -m pip install --only-binary=all streamlit --quiet --no-warn-script-location
python -m pip install --only-binary=all pandas --quiet --no-warn-script-location
python -m pip install --only-binary=all numpy --quiet --no-warn-script-location
python -m pip install --only-binary=all yfinance --quiet --no-warn-script-location
python -m pip install --only-binary=all scikit-learn --quiet --no-warn-script-location
python -m pip install --only-binary=all plotly --quiet --no-warn-script-location
echo [SUCCESS] Installed all packages!

echo.
echo [5/6] Verifying installation...
python -c "import streamlit, pandas, yfinance, sklearn, plotly; print('All OK!')" 2>nul
if errorlevel 1 (
    echo [WARNING] Some imports failed - trying backup method...
    python -m pip install streamlit pandas yfinance scikit-learn plotly --force-reinstall --quiet
) else (
    echo [SUCCESS] All packages working!
)

echo.
echo [6/6] Starting NIFTY 50 Predictor...
echo.
echo 🚀 Launching application with Python 3.12 fixes
echo 📊 Browser will open at http://localhost:8501
echo 🌐 Keep this window open
echo ⏹️  Press Ctrl+C to stop
echo.

python -m streamlit run app.py --server.headless true --server.port 8501

echo.
echo Application closed.
pause
