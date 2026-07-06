from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name: str) -> Path:
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")
    return path


def print_block(title: str, content: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(content.strip())
