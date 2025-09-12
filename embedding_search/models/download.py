import os
from huggingface_hub import snapshot_download
import time
import traceback

# Configuration
model_id = "nvidia/Cosmos-Embed1-448p"
model_dir = "./nvidia_cosmos_embed_1"

print(f"🚀 Starting download for model: {model_id}")
print(f"📂 Download location: {model_dir}")
print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

try:
    # Download the complete model
    local_dir = snapshot_download(
        repo_id="nvidia/Cosmos-Embed1-448p",
        local_dir=model_dir,
        resume_download=True,  # Resume if interrupted
        local_files_only=False
    )
    
    print(f"✅ Download completed successfully!")
    print(f"📂 Model saved to: {local_dir}")
    print(f"⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check downloaded files
    files = os.listdir(local_dir)
    total_size = 0
    for f in files:
        file_path = os.path.join(local_dir, f)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    
    print(f"📊 Downloaded {len(files)} files")
    print(f"💾 Total size: {total_size / 1024**3:.2f} GB")
    
    # List key files
    key_files = [
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'preprocessing_config.json',
        'model.safetensors.index.json'
    ]
    
    print(f"\n📋 Key files:")
    for f in key_files:
        if f in files:
            print(f"   ✅ {f}")
        else:
            print(f"   ❌ {f}")
    
    # List weight files
    weight_files = [f for f in files if f.endswith('.safetensors') and f.startswith('model-')]
    print(f"\n⚖️ Model weight files: {len(weight_files)}")
    for f in sorted(weight_files):
        size_gb = os.path.getsize(os.path.join(local_dir, f)) / 1024**3
        print(f"   💾 {f} ({size_gb:.1f} GB)")

except Exception as e:
    print(f"❌ Download failed: {e}")
    import traceback
    traceback.print_exc()
