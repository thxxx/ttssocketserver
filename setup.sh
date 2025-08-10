apt-get update
apt-get install -y ffmpeg
python3 -m pip install flask flask-cors openai elevenlabs fastapi uvicorn[standard]

# uvicorn server:app --host 0.0.0.0 --port 5000