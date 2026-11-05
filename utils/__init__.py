from importlib import import_module

_EXPORTS = {
    "AA_WEIGHTS": ".constants",
    "ALPHABET": ".constants",
    "ATOM_TYPES": ".constants",
    "ATOM_WEIGHTS": ".constants",
    "RES_ATOM14": ".constants",
    "RESTYPE_1to3": ".constants",
    "Meter_Unbound_Bound": ".docking_eval_utils",
    "rigid_transform_Kabsch_3D": ".docking_eval_utils",
    "compute_spline_curvature": ".geometry",
    "compute_spline_torsion": ".geometry",
    "construct_3d_basis": ".geometry",
    "cross_product": ".geometry",
    "normalize_vector": ".geometry",
    "parametrize_backbone_spline": ".geometry",
    "project_v2v": ".geometry",
    "randomly_rotate_and_translate": ".geometry",
    "rotate": ".geometry",
    "save_trajectory_to_pdb": ".save_structures",
    "X_N": ".so3",
    "omegas": ".so3",
    "sample": ".so3",
    "sample_vec": ".so3",
    "score_norm": ".so3",
    "score_vec": ".so3",
    "EMACallback": ".utils",
    "Features": ".utils",
    "NoiseSchedule": ".utils",
    "compute_dihedrals": ".utils",
    "get_cplx_attn_bias": ".utils",
    "get_latent_mean_std": ".utils",
    "get_resolution_indices": ".utils",
    "idx_to_seq": ".utils",
    "recursive_to": ".utils",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
