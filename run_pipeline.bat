@echo off
setlocal EnableDelayedExpansion

:: Generate a clean date string (YYYY-MM-DD) using PowerShell
for /f %%a in ('powershell -Command "Get-Date -Format yyyy-MM-dd"') do set MODEL_DATE=%%a

:: Use the date as a TAG (model:tag) which is the standard Ollama convention
set OLLAMA_MODEL_NAME=functiongemma-custom:%MODEL_DATE%
set TRAIN_DIR=functiongemma-270m-it-simple-tool-calling
set GGUF_FILE=%TRAIN_DIR%.gguf

echo ========================================================
echo  FunctionGemma End-to-End Pipeline
echo  Date: %MODEL_DATE%
echo  Target Ollama Model: %OLLAMA_MODEL_NAME%
echo ========================================================

:: 1. Environment Check
if not exist "venv" (
    echo [ERROR] Virtual environment 'venv' not found. Please run setup.bat first.
    exit /b 1
)

:: Activate venv
call venv\Scripts\activate

:: 2. Training
echo.
echo [Step 1/4] Training Model (epochs=3)...
python train.py
if %ERRORLEVEL% NEQ 0 goto :Error

:: 3. Testing
echo.
echo [Step 2/4] Testing Fine-Tuned Model...
python test_model.py
if %ERRORLEVEL% NEQ 0 goto :Error

:: 4. Conversion (HF -> GGUF)
echo.
echo [Step 3/4] Converting to GGUF format...
if not exist "llama.cpp" (
    echo [ERROR] llama.cpp directory not found. Please clone it first.
    exit /b 1
)
:: Using bf16 to match the training precision
python llama.cpp\convert_hf_to_gguf.py %TRAIN_DIR% --outfile %GGUF_FILE% --outtype bf16
if %ERRORLEVEL% NEQ 0 goto :Error

:: 5. Ollama Deployment
echo.
echo [Step 4/4] Creating Ollama Model...
echo FROM ./%GGUF_FILE% > Modelfile.%MODEL_DATE%
ollama create %OLLAMA_MODEL_NAME% -f Modelfile.%MODEL_DATE%
if %ERRORLEVEL% NEQ 0 goto :Error

:: Cleanup
del Modelfile.%MODEL_DATE%

echo.
echo ========================================================
echo  SUCCESS! Model deployed.
echo  Try it: ollama run %OLLAMA_MODEL_NAME% "Your Query"
echo ========================================================
exit /b 0

:Error
echo.
echo [ERROR] The pipeline failed at the previous step.
exit /b %ERRORLEVEL%
