import numpy as np
from pathlib import Path
from datetime import datetime

class OnlineNormalizer:

    def __init__(self, alpha=0.1, output_dir=None) -> None:
        self.ema = 0
        self.emv = 0
        self.alpha = alpha
        self.output_file = None
        self.set_output(output_dir)

    def set_output(self, output_dir):
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.output_file = Path(output_dir).joinpath(f"online_normalization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            with open(self.output_file, "w") as f:
                f.write("ema,emv\n")

    def update(self, x):
        self.ema = self.alpha * x + (1 - self.alpha) * self.ema
        self.emv = self.alpha * (x - self.ema) ** 2 + (1 - self.alpha) * self.emv
        with open(self.output_file, "a") as f:
            f.write(f"{self.ema},{self.emv}\n")

    def normalize(self, x):
        print(f"Normalizing with {self.ema} and {self.emv}")
        return (x - self.ema) / np.sqrt(self.emv + 1e-8)