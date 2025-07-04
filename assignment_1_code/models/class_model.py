import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''
        # 1. Make sure the target directory exists
        save_dir.mkdir(parents=True, exist_ok=True)

        # 2. Build filename
        fname = "model"
        if suffix:
            fname += f"_{suffix}"
        fname += ".pth"

        # 3. Full path
        file_path = save_dir / fname

        # 4. Save the state_dict of the entire module (wrapper + net)
        torch.save(self.state_dict(), file_path)
        return file_path
    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No model file found at {path}")

        # 1. Load the saved state_dict (map to CPU by default)
        state_dict = torch.load(path, map_location=torch.device('cpu'))

        # 2. Load into the current module
        self.load_state_dict(state_dict)