from data.mmshop_loader import MMShopLoader
from data.pixelrec_loader import PixelRecLoader
from data.vogue_loader import VOGUELoader

LOADER_REGISTRY = {
    "vogue": VOGUELoader,
    "pixelrec": PixelRecLoader,
    "mmshop": MMShopLoader,
}


def build_loader(dataset_name: str, root: str, **kwargs):
    if dataset_name not in LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(LOADER_REGISTRY)}")
    return LOADER_REGISTRY[dataset_name](root, **kwargs)
