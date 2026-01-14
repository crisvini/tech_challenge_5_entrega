python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install -U pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install ultralytics
