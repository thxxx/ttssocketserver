from huggingface_hub import HfApi
import os

print(os.path.exists("../../../../dataset/audios_sampled.zip"))

api = HfApi()

api.upload_file(
    path_or_fileobj="../../../../dataset/audios_sampled.zip",
    path_in_repo="audios_sampled.zip",
    repo_id="Daniel777/personals",
    repo_type="model"
)