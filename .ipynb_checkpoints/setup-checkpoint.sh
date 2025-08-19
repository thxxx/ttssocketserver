apt-get update
apt-get install -y ffmpeg
pip install --upgrade pip
pip install flask flask-cors --ignore-installed
pip install openai elevenlabs fastapi uvicorn[standard] orjson
pip install faster-whisper soundfile librosa
pip install --upgrade transformers datasets[audio] accelerate
pip install pydub

git config user.email zxcv05999@naver.com
git config user.name thxxx

pip install -U nemo_toolkit['asr']

# uvicorn server:app --host 0.0.0.0 --port 5000