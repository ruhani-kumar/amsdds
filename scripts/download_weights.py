import os
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = os.environ.get("HF_REPO_ID", "ruhanikumar/amsdds-weights")
HF_TOKEN = os.environ.get("HF_TOKEN")

WEIGHTS_DIR = Path("backend/ml_core/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "layer1_multimodal_mobilenetv3.pt",
    "layer2_panderm_ft_v1.pt",
    "ood_scores_mobilenet.npz",
]


def main() -> None:
    print(f"Downloading model weights from {REPO_ID}...")

    for filename in FILES:
        print(f"Downloading {filename}...")

        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=HF_TOKEN,
            local_dir=WEIGHTS_DIR,
        )

        print(f"Downloaded: {path}")

    print("All model weights downloaded successfully.")


if __name__ == "__main__":
    main()