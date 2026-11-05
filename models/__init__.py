__all__ = ["AllAtomEnergyModel"]


def __getattr__(name):
    if name != "AllAtomEnergyModel":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .energy import AllAtomEnergyModel

    globals()[name] = AllAtomEnergyModel
    return AllAtomEnergyModel
