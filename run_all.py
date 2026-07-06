"""一次运行 8 个算法 Demo。

说明：为了避免每个脚本单独启动 Python 导致 PyTorch 重复初始化，这里直接用 importlib 加载并调用 main()。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demos"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

DEMOS = [
    "01_cnn_demo.py",
    "02_yolo_toy_demo.py",
    "03_density_map_demo.py",
    "04_rnn_lstm_demo.py",
    "05_transformer_attention_demo.py",
    "06_autoencoder_demo.py",
    "07_gan_demo.py",
    "08_diffusion_demo.py",
]


def run_script(path: Path) -> None:
    module_name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(f"{path} has no main()")
    module.main()


def main() -> None:
    for demo in DEMOS:
        path = ROOT / "demos" / demo
        print(f"\n========== Running {demo} ==========")
        run_script(path)

    try:
        import generate_principle_cards
        print("\n========== Generating Chinese principle cards ==========")
        generate_principle_cards.main()
    except Exception as e:
        print(f"生成中文详细原理图时发生错误: {e}")

    print("\n全部 Demo 运行完成，图片已输出到 outputs/ 目录。")


if __name__ == "__main__":
    main()
