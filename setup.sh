apt-get update
apt-get install -y ffmpeg
pip install --upgrade pip
pip install flask flask-cors --ignore-installed
pip install openai fastapi uvicorn[standard] orjson
pip install faster-whisper soundfile librosa
pip install --upgrade transformers datasets[audio] accelerate
pip install pydub

git config user.email zxcv05999@naver.com
git config user.name thxxx

# sudo apt-get install -y python3.10-dev build-essential
pip install -U nemo_toolkit['asr']

pip install --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html
pip install torchaudio numpy lhotse huggingface_hub safetensors tensorboard vocos
pip install cn2an inflect s3tokenizer diffusers conformer pkuseg pykakasi resemble-perth

# Tokenization
pip install jieba piper_phonemize pypinyin
pip install "setuptools<81"

# pip install k2==1.24.4.dev20250807+cuda12.8.torch2.8.0 -f https://k2-fsa.github.io/k2/cuda.html
# pip install k2==1.24.4.dev20240211+cuda12.1.torch2.2.0 -f https://k2-fsa.github.io/k2/cuda.html
# pip uninstall -y torchaudio
# pip install torchaudio

pip install vllm
pip install --upgrade "pyzmq<26"

# uvicorn yourSpeakerServer:app --host 0.0.0.0 --port 5000