import math

import numpy as np
from scipy.linalg import det, svd
from sklearn.neighbors import BallTree


def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        SS = np.diag([1.0, 1.0, -1.0])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t


class Meter_Unbound_Bound(object):
    def __init__(self):

        self.complex_rmsd_list = []
        self.ligand_rmsd_list = []
        self.receptor_rmsd_list = []
        self.interface_rmsd_list = []
        self.fnat_list = []
        self.dockQ_list = []

        self.complex_rmsd_list_all = []
        self.ligand_rmsd_list_all = []
        self.receptor_rmsd_list_all = []
        self.interface_rmsd_list_all = []
        self.fnat_list_all = []
        self.dockQ_list_all = []

        self.bad_case = 0

    def compute_lrmsd(
        self,
        ligand_coors_pred,
        receptor_coors_pred,
        ligand_coors_true,
        receptor_coors_true,
    ):
        ligand_coors_pred = ligand_coors_pred.detach().cpu().numpy()
        receptor_coors_pred = receptor_coors_pred.detach().cpu().numpy()

        ligand_coors_true = ligand_coors_true.detach().cpu().numpy()
        receptor_coors_true = receptor_coors_true.detach().cpu().numpy()

        ligand_rmsd = np.sqrt(
            np.mean(np.sum((ligand_coors_pred - ligand_coors_true) ** 2, axis=1))
        )

        return ligand_rmsd

    def update_rmsd(
        self,
        ligand_coors_pred,
        receptor_coors_pred,
        ligand_coors_true,
        receptor_coors_true,
        at_k=1,
    ):
        ligand_coors_pred = ligand_coors_pred.detach().cpu().numpy()
        receptor_coors_pred = receptor_coors_pred.detach().cpu().numpy()

        ligand_coors_true = ligand_coors_true.detach().cpu().numpy()
        receptor_coors_true = receptor_coors_true.detach().cpu().numpy()

        ligand_rmsd = np.sqrt(
            np.mean(np.sum((ligand_coors_pred - ligand_coors_true) ** 2, axis=1))
        )
        receptor_rmsd = np.sqrt(
            np.mean(np.sum((receptor_coors_pred - receptor_coors_true) ** 2, axis=1))
        )

        complex_coors_pred = np.concatenate(
            (ligand_coors_pred, receptor_coors_pred), axis=0
        )
        complex_coors_true = np.concatenate(
            (ligand_coors_true, receptor_coors_true), axis=0
        )

        R, b = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
        complex_coors_pred_aligned = ((R @ complex_coors_pred.T) + b).T

        complex_rmsd = np.sqrt(
            np.mean(
                np.sum((complex_coors_pred_aligned - complex_coors_true) ** 2, axis=1)
            )
        )

        interface_rmsd = self.compute_IRMSD(
            ligand_coors_pred,
            receptor_coors_pred,
            ligand_coors_true,
            receptor_coors_true,
        )

        fnat = self.compute_Fnat(
            ligand_coors_pred,
            receptor_coors_pred,
            ligand_coors_true,
            receptor_coors_true,
        )
        dockq = self.compute_DockQ(fnat, interface_rmsd, ligand_rmsd)

        self.complex_rmsd_list_all.append(complex_rmsd)
        self.ligand_rmsd_list_all.append(ligand_rmsd)
        self.receptor_rmsd_list_all.append(receptor_rmsd)
        self.interface_rmsd_list_all.append(interface_rmsd)
        self.fnat_list_all.append(fnat)
        self.dockQ_list_all.append(dockq)

        if complex_rmsd < 50:
            self.complex_rmsd_list.append(complex_rmsd)
            self.ligand_rmsd_list.append(ligand_rmsd)
            self.receptor_rmsd_list.append(receptor_rmsd)
            self.interface_rmsd_list.append(interface_rmsd)
            self.fnat_list.append(fnat)
            self.dockQ_list.append(dockq)
        else:
            self.bad_case += 1

        return ligand_rmsd, interface_rmsd, complex_rmsd, dockq

    def compute_CRMSD(self, pcd1, pcd2):
        rot_mat, transl_vec = self.Kabsch(pcd1, pcd2)
        pcd1_align = pcd1 @ rot_mat + transl_vec
        return np.sqrt(np.mean(np.sum((pcd1_align - pcd2) ** 2, axis=1)))

    def compute_IRMSD(self, lig, rec, lig_gt, rec_gt, iface_cutoff=8.0):

        bt1 = BallTree(lig_gt)
        dist1, _ = bt1.query(rec_gt, k=1)
        rec_iface_gt = rec_gt[np.where(dist1 < iface_cutoff)[0]]
        rec_iface = rec[np.where(dist1 < iface_cutoff)[0]]
        bt2 = BallTree(rec_gt)
        dist2, _ = bt2.query(lig_gt, k=1)
        lig_iface_gt = lig_gt[np.where(dist2 < iface_cutoff)[0]]
        lig_iface = lig[np.where(dist2 < iface_cutoff)[0]]

        return self.compute_CRMSD(
            np.vstack([lig_iface, rec_iface]), np.vstack([lig_iface_gt, rec_iface_gt])
        )

    def summarize(self, reduction_rmsd="median", include_failed=False):
        """
        Summarize RMSD metrics.

        Args:
            reduction_rmsd: 'mean', 'median', or 'std'
            include_failed: If True, include failed cases in statistics

        Returns:
            Tuple of (ligand_rmsd, complex_rmsd, interface_rmsd, bad_case_count)
        """

        if include_failed:
            complex_rmsd_list = self.complex_rmsd_list_all
            ligand_rmsd_list = self.ligand_rmsd_list_all
            interface_rmsd_list = self.interface_rmsd_list_all
        else:
            complex_rmsd_list = self.complex_rmsd_list
            ligand_rmsd_list = self.ligand_rmsd_list
            interface_rmsd_list = self.interface_rmsd_list

        if len(complex_rmsd_list) == 0:
            return 0.0, 0.0, 0.0, self.bad_case

        if reduction_rmsd == "mean":
            complex_rmsd_summarized = np.mean(complex_rmsd_list)
            ligand_rmsd_summarized = np.mean(ligand_rmsd_list)
            interface_rmsd_summarized = np.mean(interface_rmsd_list)
        elif reduction_rmsd == "median":
            complex_rmsd_summarized = np.median(complex_rmsd_list)
            ligand_rmsd_summarized = np.median(ligand_rmsd_list)
            interface_rmsd_summarized = np.median(interface_rmsd_list)
        elif reduction_rmsd == "std":
            complex_rmsd_summarized = np.std(complex_rmsd_list)
            ligand_rmsd_summarized = np.std(ligand_rmsd_list)
            interface_rmsd_summarized = np.std(interface_rmsd_list)
        else:
            raise ValueError("Meter_Unbound_Bound: reduction_rmsd mis specified!")

        return (
            ligand_rmsd_summarized,
            complex_rmsd_summarized,
            interface_rmsd_summarized,
            self.bad_case,
        )

    def dockQ(self, include_failed=False):
        """
        Calculate DockQ statistics.

        Args:
            include_failed: If True, include failed cases in statistics

        Returns:
            Tuple of (mean, median, std) for DockQ scores
        """
        dockq_list = self.dockQ_list_all if include_failed else self.dockQ_list

        if len(dockq_list) == 0:
            return 0.0, 0.0, 0.0

        dockq_array = np.array(dockq_list)
        return np.mean(dockq_array), np.median(dockq_array), np.std(dockq_array)

    def get_summary_stats(self, reduction_rmsd="median"):
        """
        Get comprehensive summary statistics for both scenarios.

        Returns:
            Dictionary with stats for both excluding and including failed cases
        """
        stats = {}

        ligand_rmsd, complex_rmsd, interface_rmsd, bad_cases = self.summarize(
            reduction_rmsd=reduction_rmsd, include_failed=False
        )
        dockq_mean, dockq_median, dockq_std = self.dockQ(include_failed=False)

        stats["excluding_failed"] = {
            "ligand_rmsd": ligand_rmsd,
            "complex_rmsd": complex_rmsd,
            "interface_rmsd": interface_rmsd,
            "dockq_mean": dockq_mean,
            "dockq_median": dockq_median,
            "dockq_std": dockq_std,
            "n_samples": len(self.complex_rmsd_list),
            "bad_cases": bad_cases,
        }

        ligand_rmsd_all, complex_rmsd_all, interface_rmsd_all, _ = self.summarize(
            reduction_rmsd=reduction_rmsd, include_failed=True
        )
        dockq_mean_all, dockq_median_all, dockq_std_all = self.dockQ(
            include_failed=True
        )

        stats["including_failed"] = {
            "ligand_rmsd": ligand_rmsd_all,
            "complex_rmsd": complex_rmsd_all,
            "interface_rmsd": interface_rmsd_all,
            "dockq_mean": dockq_mean_all,
            "dockq_median": dockq_median_all,
            "dockq_std": dockq_std_all,
            "n_samples": len(self.complex_rmsd_list_all),
            "bad_cases": bad_cases,
        }

        return stats

    def compute_Fnat(self, lig, rec, lig_gt, rec_gt, iface_cutoff=8.0):
        bt1_gt = BallTree(lig_gt)
        dist1_gt, _ = bt1_gt.query(rec_gt, k=1)
        rec_iface_ind_gt = np.where(dist1_gt < iface_cutoff)[0]
        lig_iface_ind_gt = bt1_gt.query_radius(rec_gt[rec_iface_ind_gt], iface_cutoff)
        iface_pair_gt = set(
            [
                (i, j)
                for n, i in enumerate(rec_iface_ind_gt)
                for j in lig_iface_ind_gt[n]
            ]
        )

        bt1 = BallTree(lig)
        dist1, _ = bt1.query(rec, k=1)
        rec_iface_ind = np.where(dist1 < iface_cutoff)[0]
        if len(rec_iface_ind) == 0:
            return 0.0

        lig_iface_ind = bt1.query_radius(rec[rec_iface_ind], iface_cutoff)
        iface_pair = set(
            [(i, j) for n, i in enumerate(rec_iface_ind) for j in lig_iface_ind[n]]
        )

        return len(iface_pair & iface_pair_gt) / len(iface_pair_gt)

    def compute_DockQ(self, Fnat, IRMSD, LRMSD):
        return (
            Fnat + 1.0 / (1.0 + (IRMSD / 1.5) ** 2) + 1.0 / (1.0 + (LRMSD / 8.5) ** 2)
        ) / 3.0

    def Kabsch(self, Y1, Y2, normal1=None, normal2=None):

        Y1_mean = Y1.mean(axis=0, keepdims=True)
        Y2_mean = Y2.mean(axis=0, keepdims=True)

        if normal1 is not None and normal2 is not None:
            A = np.vstack([Y1 - Y1_mean, normal1]).T @ np.vstack(
                [Y2 - Y2_mean, normal2]
            )
        else:
            A = (Y1 - Y1_mean).T @ (Y2 - Y2_mean)
        U, _, Vt = svd(A)
        d = np.sign(det(U @ Vt))

        corr_mat = np.diag(np.asarray([1, 1, d]))

        rot_mat = U @ corr_mat @ Vt
        transl_vec = Y2_mean - Y1_mean @ rot_mat

        return rot_mat, transl_vec
