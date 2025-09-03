from huggingface_hub import HfApi
import os

print(os.path.exists("/Users/gimhojin/Desktop/projects/eidos/matcha_ljspeech.ckpt"))

api = HfApi()

api.upload_file(
    path_or_fileobj="/Users/gimhojin/Desktop/projects/eidos/matcha_ljspeech.ckpt",
    path_in_repo="matcha_ljspeech.ckpt",
    repo_id="Daniel777/personals",
    repo_type="model"
)