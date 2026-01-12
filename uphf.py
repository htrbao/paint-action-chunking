from huggingface_hub import HfApi
from dotenv import load_dotenv
load_dotenv()

repo_id = "ducido/libero_s100_baseline_decay300k"
local_folder = "/data/outputs/train/2025-09-18/08-24-29_libero_100%_test"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Initial commit",
    # optional filters
    ignore_patterns=["**/__pycache__/**", "**/*.tmp", "**/.ipynb_checkpoints/**"],
    # allow_patterns=["**/*.pt","**/*.json"]  # alternatively, whitelist
)