from huggingface_hub import snapshot_download

model_path = snapshot_download(
    "BAAI/bge-m3",
    local_dir="../models/bge-m3"
)

print(f"Model downloaded to: {model_path}")