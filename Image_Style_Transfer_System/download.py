import sys
from pathlib import Path
from datasets.utils.logging import enable_progress_bar
from huggingface_hub import snapshot_download

enable_progress_bar()

local_dataset_path = Path(
    "/Alpha/College_new/Production internship/2ImageStyleTransferSystem/assets/style_expert"
)
local_dataset_path.mkdir(parents=True, exist_ok=True)

print(f"目标目录: {local_dataset_path}")
print("开始下载（自动显示 tqdm 进度条）...\n")

try:
    snapshot_download(
        repo_id="HH-LG/StyleExpert",
        repo_type="dataset",
        local_dir=str(local_dataset_path),
        max_workers=4,
    )
except Exception as e:
    print(f"❌ 下载失败: {e}")
    sys.exit(1)

print(f"\n✅ 下载完成: {local_dataset_path}")
for p in sorted(local_dataset_path.rglob("*")):
    if p.is_file():
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.relative_to(local_dataset_path)}  ({size_mb:.1f} MB)")