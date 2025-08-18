@echo off
REM =========================================
REM Batch file to launch SSL Streamlit Dashboard
REM =========================================

REM 1. Change this to your environment folder name
SET ENV_NAME=LABS

REM 2. Change this to your project path
SET PROJECT_DIR=D:\EWU\10th Semester\CSE475\LABS\Project\Streamlit App

REM 3. Activate the environment
CALL "%PROJECT_DIR%\%ENV_NAME%\Scripts\activate.bat"

REM 4. Install Streamlit if missing (optional, uncomment if needed)
REM pip install --upgrade pip
REM pip install streamlit ultralytics torch torchvision pandas matplotlib pillow

REM 5. Run Streamlit dashboard
streamlit run "%PROJECT_DIR%\SSL_APP.py" --server.port 8501

REM 6. Pause to see any messages after closing
pause
