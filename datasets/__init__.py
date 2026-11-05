from importlib import import_module

__all__ = [
    "ProteinDockingDataset",
    "build_dataset",
    "build_dataloader",
    "make_protein_collate_fn",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".dataset", __name__), name)
    globals()[name] = value
    return value
