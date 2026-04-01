from datasets_adapters import (
    AI2DAdapter,
    HDEpicAdapter,
    LocalManifestAdapter,
    MathVistaAdapter,
    MLVUAdapter,
    MMMUAdapter,
    OCRBenchAdapter,
    VideoMMEAdapter,
)


_ADAPTERS = {
    "local_manifest": LocalManifestAdapter,
    "hd_epic": HDEpicAdapter,
    "mmmu": MMMUAdapter,
    "mathvista": MathVistaAdapter,
    "ocrbench": OCRBenchAdapter,
    "ai2d": AI2DAdapter,
    "videomme": VideoMMEAdapter,
    "mlvu": MLVUAdapter,
}


def list_adapters():
    return sorted(_ADAPTERS)


def get_adapter(name):
    key = name.lower()
    if key not in _ADAPTERS:
        raise KeyError(f"Unknown dataset adapter `{name}`. Available: {', '.join(list_adapters())}")
    return _ADAPTERS[key]()
