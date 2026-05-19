"""Model loading utilities with HuggingFace Hub auto-download."""
import os
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import torch
from tqdm.auto import tqdm


MODEL_URL = "https://huggingface.co/henryhs/rag-table/resolve/main/unet_best.pt"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ragtab"


def get_cache_dir() -> Path:
    """Get cache directory, override-able via RAGTAB_CACHE_DIR env var."""
    cache_dir = Path(os.environ.get("RAGTAB_CACHE_DIR", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class _TqdmDownloadProgress(tqdm):
    """tqdm hook for urlretrieve."""
    def update_to(self, block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def download_model(force: bool = False) -> Path:
    """
    Download model from HuggingFace Hub if not cached.
    
    Args:
        force: Re-download even if cached file exists.
    
    Returns:
        Path to the downloaded model file.
    """
    cache_dir = get_cache_dir()
    model_path = cache_dir / "unet_best.pt"
    
    if model_path.exists() and not force:
        return model_path
    
    print(f"📥 Downloading RagTable model from HuggingFace...")
    
    try:
        with _TqdmDownloadProgress(
            unit='B', unit_scale=True, unit_divisor=1024,
            miniters=1, desc="unet_best.pt", ncols=80
        ) as t:
            urlretrieve(MODEL_URL, model_path, reporthook=t.update_to)
    except (URLError, Exception) as e:
        if model_path.exists():
            model_path.unlink()  
        raise RuntimeError(
            f"Failed to download model from {MODEL_URL}\n"
            f"Error: {e}\n"
            f"You can manually download and place it at: {model_path}"
        )
    
    return model_path

_MODEL_CACHE = {}
def load_model(device=None, model_path=None, compile_mode=False, use_cache=True):

    """
    Load EfficientUNet model, auto-downloading if needed.
    
    Args:
        device: torch device. Auto-detects CUDA if None.
        model_path: Custom model path. Auto-downloads from HF if None.
        compile_mode: Use torch.compile for speedup (PyTorch 2.0+).
    
    Returns:
        (model, device) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = (str(model_path), str(device), compile_mode)

    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], device
    
    if model_path is None:
        model_path = download_model()
    
    from .model import EfficientUNet
    model = EfficientUNet(out_ch=5, pretrained=False).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    if compile_mode and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception:
            pass

    if use_cache:
        _MODEL_CACHE[cache_key] = model
    
    return model, device