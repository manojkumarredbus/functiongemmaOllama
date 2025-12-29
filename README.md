# FunctionGemma to Ollama Pipeline 🚀

A streamlined, end-to-end workflow to fine-tune Google's **FunctionGemma** models (specialized for tool calling/function calling), convert them to GGUF format, and deploy them directly to **Ollama** for local inference.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Ollama](https://img.shields.io/badge/Ollama-Deployed-black)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

*   **Custom Fine-Tuning**: Uses Hugging Face `trl` and `SFTTrainer` to fine-tune FunctionGemma on your own tool definitions and query samples.
*   **Automated Conversion**: Seamlessly converts the fine-tuned model to GGUF using `llama.cpp`.
*   **Ollama Deployment**: Automatically creates a custom Ollama model with a date-stamped tag (e.g., `functiongemma-custom:2025-12-29`).
*   **Windows Optimized**: Includes batch scripts for one-click setup and execution, with fixes for common Windows SSL and Path issues.
*   **GPU Accelerated**: Pre-configured for NVIDIA GPUs (BF16/FP16 support enabled).

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
2.  **Git**: [Download Git](https://git-scm.com/)
3.  **Ollama**: [Download Ollama](https://ollama.com/)
4.  **NVIDIA GPU (Recommended)**: With latest drivers and CUDA support.

## 📦 Installation

1.  **Clone this repository**:
    ```bash
    git clone https://github.com/manojkumarredbus/functiongemmaOllama.git
    cd functiongemmaOllama
    ```

2.  **Run the Setup Script**:
    Double-click `setup.bat` or run:
    ```cmd
    .\setup.bat
    ```
    *This creates a virtual environment and installs all necessary Python dependencies.*

3.  **Clone llama.cpp** (Required for GGUF conversion):
    The pipeline expects `llama.cpp` to be in the root directory.
    ```bash
    git clone https://github.com/ggerganov/llama.cpp.git
    ```
    *Note: You may need to install llama.cpp dependencies via `venv\Scripts\pip install -r llama.cpp\requirements.txt` if they aren't covered by the main setup, though the pipeline mainly uses the conversion script.*

## 🚀 Usage

### One-Click Pipeline

The magic happens in `run_pipeline.bat`. This script performs the entire workflow in sequence:

```cmd
.\run_pipeline.bat
```

**What this script does:**
1.  **Trains** the model using `train.py` (Default: 3 epochs).
2.  **Tests** the Python model locally with `test_model.py`.
3.  **Converts** the model to GGUF format using `llama.cpp`.
4.  **Imports** the model into Ollama as `functiongemma-custom:YYYY-MM-DD`.

### Manual Execution

If you prefer to run steps manually:

1.  **Activate Environment**:
    ```cmd
    venv\Scripts\activate
    ```

2.  **Train**:
    ```cmd
    python train.py
    ```

3.  **Test (Python)**:
    ```cmd
    python test_model.py
    ```

4.  **Convert to GGUF**:
    ```cmd
    python llama.cpp\convert_hf_to_gguf.py functiongemma-270m-it-simple-tool-calling --outfile model.gguf --outtype bf16
    ```

5.  **Import to Ollama**:
    ```cmd
    echo FROM ./model.gguf > Modelfile
    ollama create my-function-model -f Modelfile
    ```

## 🧪 Testing with Ollama

Once deployed, you can run the model directly:

```cmd
ollama run functiongemma-custom:2025-12-29 "What is the reimbursement limit for travel meals?"
```

*Note: FunctionGemma expects specific XML-like tags for tool definitions. Using `ollama run` interactively might not yield perfect tool calls without the proper system prompt context, but the model is trained to recognize the structure.*

## ⚙️ Customization

To train on **your own data**:

1.  Open `train.py`.
2.  Locate the `simple_tool_calling` list.
3.  Add your own examples following the format:
    ```python
    {
        "user_content": "Your user query here",
        "tool_name": "function_to_call",
        "tool_arguments": "{\"arg_name\": \"value\"}"
    }
    ```
4.  Update the `TOOLS` definition list if you are adding new function schemas.
5.  Re-run `run_pipeline.bat`.

## 📝 Notes

*   **SSL Verification**: The `train.py` script includes patches to bypass SSL verification errors common in some corporate Windows environments. Do not use this configuration in a production security-sensitive environment without reviewing `train.py`.
*   **Hardware**: The default config is optimized for an RTX 4090 (BF16 enabled). If you are on an older GPU, you may need to edit `train.py` to set `bf16=False` and `fp16=True`.

## 🤝 Credits

*   [Google FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma)
*   [Ollama](https://ollama.com/)
*   [llama.cpp](https://github.com/ggerganov/llama.cpp)
