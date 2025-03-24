# Orpheus-GGUF
[Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) inference using python bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Installation
Clone the repo:
```sh
git clone https://github.com/zuellni/orpheus-gguf
cd orpheus-gguf
```

Create a venv:
```sh
python -m venv venv
venv\scripts\activate # windows
source venv/bin/activate # linux
```

Install torch:
```sh
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Install llama-cpp-python:
```sh
pip install llama-cpp-python -C cmake.args="-DGGML_CUDA=1;-DGGML_CUDA_F16=1;-DGGML_CUDA_FA_ALL_QUANTS=1"
```
See the instructions in the [original repo](https://github.com/abetlen/llama-cpp-python) if this fails.

Install other requirements:
```sh
pip install -r requirements.txt
```
