import os
os.environ["HF_HOME"] = "/mnt/lustre-grete/usr/u12045/vla/hf_cache"
import time
import requests
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

def download_to_hf_cache(
    repo_id: str,
    max_retries: int = 5,
    sleep_between_retries: float = 8.0,
    throttle_delay: float = 0.5,
    repo_revision: str = "main",  # Optional: commit hash, branch, or tag
):
    print(f"📦 Listing all files from {repo_id}@{repo_revision}...")
    all_files = list_repo_files(repo_id, repo_type="dataset", revision=repo_revision)

    print(f"🔍 Found {len(all_files)} files (no extension filtering).")

    for file in tqdm(all_files, desc="📥 Downloading to HF cache"):
        for attempt in range(max_retries):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    repo_type="dataset",
                    revision=repo_revision,
                    local_dir=None,  # Store in Hugging Face cache
                    local_dir_use_symlinks=False  # Force full download, not symlink
                )
                break  # success
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"⚠️ Rate limited on {file}, sleeping {sleep_between_retries}s...")
                else:
                    print(f"❌ HTTP error on {file}: {e}")
                time.sleep(sleep_between_retries)
            except Exception as e:
                print(f"❌ Unexpected error on {file}: {e}")
                time.sleep(sleep_between_retries)
        time.sleep(throttle_delay)  # Throttle between downloads

    print("✅ All files downloaded into Hugging Face cache.")

# ========= USAGE =========
if __name__ == "__main__":
    repo_id = "ducido/h5_calvin"
    download_to_hf_cache(repo_id)