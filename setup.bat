@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete.
echoTo run training: python train.py
echo To run inference: python test_model.py
pause
