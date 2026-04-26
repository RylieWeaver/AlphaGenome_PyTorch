from huggingface_hub import HfApi

repo_id = "RylieWeaver/alphagenome-pytorch"
local_path = "alphagenome_converted_state_dict.pt"

api = HfApi()

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,
    exist_ok=True,
)

api.upload_file(
    path_or_fileobj=local_path,
    path_in_repo="alphagenome_converted_state_dict.pt",
    repo_id=repo_id,
    repo_type="model",
)

print(f"Uploaded {local_path} to https://huggingface.co/{repo_id}")
