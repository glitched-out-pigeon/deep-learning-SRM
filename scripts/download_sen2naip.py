from pathlib import Path
from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sen2naip"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SEN2NAIP REAL DATASET DOWNLOAD")
print("=" * 70)

print("\nDownloading cross-sensor dataset...")

path = hf_hub_download(
    repo_id="isp-uv-es/SEN2NAIP",
    repo_type="dataset",
    filename="cross-sensor/cross-sensor.zip",
    local_dir=str(OUTPUT_DIR),
)

print("\n✅ Download complete")
print("File:", path)
print("Size:", Path(path).stat().st_size / (1024 ** 3), "GB")