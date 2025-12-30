@echo off
setlocal EnableDelayedExpansion

echo ========================================================
echo  Jlama (Maven) Pipeline
echo ========================================================

set PROJECT_ROOT=%CD%
set RUNNER_DIR=jlama_runner
set TRAIN_DIR=..\functiongemma-270m-it-simple-tool-calling
set JLAMA_MODEL_DIR=..\functiongemma-270m-it-simple-tool-calling-jlama

cd %RUNNER_DIR%

:: 1. Build (Download dependencies)
echo [INFO] Building Jlama Runner...
call mvn clean compile
if %ERRORLEVEL% NEQ 0 exit /b 1

:: 2. Quantize (if needed)
if not exist "%JLAMA_MODEL_DIR%" (
    echo [INFO] Quantizing model for Jlama (Q8)...
    :: Pass flags for Vector API
    set MAVEN_OPTS=--add-modules jdk.incubator.vector --enable-preview
    
    call mvn exec:java -Dexec.mainClass="com.rblab.RunJlama" -Dexec.args="quantize %TRAIN_DIR% %JLAMA_MODEL_DIR% Q8"
)

:: 3. Run REST API
echo.
echo [INFO] Starting Jlama REST API...
echo Access at: http://localhost:8080
echo.

set MAVEN_OPTS=--add-modules jdk.incubator.vector --enable-preview
call mvn exec:java -Dexec.mainClass="com.rblab.RunJlama" -Dexec.args="restapi %JLAMA_MODEL_DIR%"

cd %PROJECT_ROOT%
exit /b 0
