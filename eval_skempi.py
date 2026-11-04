import torch
import pdb as debugger
import numpy as np
import scipy
import random
import glob
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve
# %env TORCH_EXTENSIONS_DIR=.
from models import AllAtomEnergyModel, FAEnergyModel
from datasets import ProteinDataset
from utils import load_esm_embedding

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})



def eval_skempi(args, gpu, ckpt_path, model_type, save_dir, test_data=None, embedding=None):
    device = torch.device(f"cuda:{gpu}")
    model_ckpt, _, model_args = torch.load(ckpt_path, map_location=device)
    # model_args.encoder = "sa"
    # model_args.normalization = "layer"
    model_args.noise_scaler = args.noise_scaler

    if model_type == "all_atom":
        model = AllAtomEnergyModel(model_args, device=device).cuda(device=device)
    elif model_type == "fa":
        model = FAEnergyModel(model_args, device=device).cuda(device=device)

    model.load_state_dict(model_ckpt)
    model.eval()

    if test_data is None:
        test_data = ProteinDataset("data/skempi/skempi_all.pkl", 50)

    if embedding is None:
        try:
            embedding = torch.load("data/skempi/skempi_all.pt")
        except FileNotFoundError:
            embedding = load_esm_embedding(test_data.data, ['binder_full', 'target_full'], device=device)
            torch.save(embedding, "data/skempi/skempi_all.pt")

    # with torch.no_grad():
    rmsd_history_bnd_tgt = []
    min_rot_backbone_history_bnd_tgt = []
    min_rot_sc_history_bnd_tgt = []
    rmsd_after_history_bnd_tgt = []

    rmsd_history_tgt_bnd = []
    min_rot_backbone_history_tgt_bnd = []
    min_rot_sc_history_tgt_bnd = []
    rmsd_after_history_tgt_bnd = []

    wt_map = {}
    pred, label = [], []
    with torch.no_grad():
        for entry in test_data.data:
            pdb, mutation, ddg = entry['pdb']
            # entries = test_data.data[:8]
            entries = [entry]
            if len(mutation) == 0:
                binder, target = ProteinDataset.make_local_batch(entries, embedding, model_args, 'binder', 'target', device=device)
                try:
                    wt_map[pdb] = model.predict(binder, target) + model.predict(target, binder)
                except:
                    debugger.set_trace()
                # binder_pred = model.langevin_dynamics(binder, target, 100, 0.02, 0.5, len(entries))

                # rmsd, min_rot_backbone, min_rot_sc, rmsd_backbone, rmsd_sc, rmsd_after = model.eval_structures(binder, binder_pred)
                # rmsd_history_bnd_tgt.append(rmsd)
                # min_rot_backbone_history_bnd_tgt.append(min_rot_backbone)
                # min_rot_sc_history_bnd_tgt.append(min_rot_sc)
                # rmsd_after_history_bnd_tgt.append(rmsd_after)

                # target_pred = model.langevin_dynamics(target, binder, 100, 0.02, 0.5, len(entries))
                # rmsd, min_rot_backbone, min_rot_sc, rmsd_backbone, rmsd_sc, rmsd_after = model.eval_structures(target, target_pred)
                # rmsd_history_tgt_bnd.append(rmsd)
                # min_rot_backbone_history_tgt_bnd.append(min_rot_backbone)
                # min_rot_sc_history_tgt_bnd.append(min_rot_sc)
                # rmsd_after_history_tgt_bnd.append(rmsd_after)

                model.zero_grad()
        
    # # save the rmsd, min_rot_backbone, min_rot_sc histories in .pt files
    # torch.save(rmsd_history_bnd_tgt, "rmsd_history_bnd_tgt.pt")
    # torch.save(min_rot_backbone_history_bnd_tgt, "min_rot_backbone_history_bnd_tgt.pt")
    # torch.save(min_rot_sc_history_bnd_tgt, "min_rot_sc_history_bnd_tgt.pt")
    # torch.save(rmsd_history_tgt_bnd, "rmsd_history_tgt_bnd.pt")
    # torch.save(min_rot_backbone_history_tgt_bnd, "min_rot_backbone_history_tgt_bnd.pt")
    # torch.save(min_rot_sc_history_tgt_bnd, "min_rot_sc_history_tgt_bnd.pt")
    # torch.save(rmsd_after_history_bnd_tgt, "rmsd_after_history_bnd_tgt.pt")
    # torch.save(rmsd_after_history_tgt_bnd, "rmsd_after_history_tgt_bnd.pt")
    
    # # plot the RMSD for binder -> target and target -> binder
    # plt.figure()
    # plt.hist(rmsd_history_bnd_tgt, bins=100, alpha=0.5, label='binder -> target')
    # plt.hist(rmsd_history_tgt_bnd, bins=100, alpha=0.5, label='target -> binder')
    # plt.xlabel('RMSD')
    # plt.legend()
    # plt.title('RMSD distribution for binder -> target and target -> binder')

    # plt.savefig(f"rmsd_{args.noise_scaler}.png")

    # # plot the min_rot_backbone for binder -> target and target -> binder
    # plt.figure()
    # plt.hist(min_rot_backbone_history_bnd_tgt, bins=100, alpha=0.5, label='binder -> target')
    # plt.hist(min_rot_backbone_history_tgt_bnd, bins=100, alpha=0.5, label='target -> binder')
    # plt.xlabel('min_rot_backbone')
    # plt.legend()
    # plt.title('min_rot_backbone distribution for binder -> target and target -> binder')
    # plt.savefig(f"min_rot_backbone_{args.noise_scaler}.png")

    # # plot the min_rot_sc for binder -> target and target -> binder
    # plt.figure()
    # plt.hist(min_rot_sc_history_bnd_tgt, bins=100, alpha=0.5, label='binder -> target')
    # plt.hist(min_rot_sc_history_tgt_bnd, bins=100, alpha=0.5, label='target -> binder')
    # plt.xlabel('min_rot_sc')
    # plt.legend()
    # plt.title('min_rot_sc distribution for binder -> target and target -> binder')
    # plt.savefig(f"min_rot_sc_{args.noise_scaler}.png")

    # # plot the rmsd_after for binder -> target and target -> binder
    # plt.figure()
    # plt.hist(rmsd_after_history_bnd_tgt, bins=100, alpha=0.5, label='binder -> target')
    # plt.hist(rmsd_after_history_tgt_bnd, bins=100, alpha=0.5, label='target -> binder')
    # plt.title('rmsd distribution for binder -> target and target -> binder')
    # plt.xlabel('rmsd')
    # plt.legend()
    # plt.savefig(f"rmsd_after_{args.noise_scaler}.png")

    # debugger.set_trace()


    with torch.no_grad():
        for entry in test_data.data:
            pdb, mutation, ddg = entry['pdb']
            if len(mutation) > 0:
                binder, target = ProteinDataset.make_local_batch([entry], embedding, model_args, 'binder', 'target', device=device)
                score = model.predict(binder, target) + model.predict(target, binder)
                score = score - wt_map[pdb]
                pred.append(-1.0 * score.item())
                label.append(ddg)

    test_spearman = scipy.stats.spearmanr(pred, label)[0]
    test_pearson = scipy.stats.pearsonr(pred, label)[0]
    sns.regplot(x=pred, y=label)
    plt.xlabel('Predicted DDG')
    plt.ylabel('Experimental DDG')
    plt.title(f'SKEMPI test correlation: Spearman R = {test_spearman:.4f}, Pearson R = {test_pearson:.4f}')

    fname = ckpt_path.split('/')[-1]

    try:
        corr_df = pd.read_csv(f"{save_dir}/corr.csv")
    except FileNotFoundError:
        corr_df = pd.DataFrame(columns=["model", "spearman", "pearson"])

    new_entry = {"model": fname, "spearman": test_spearman, "pearson": test_pearson}
    new_df = pd.DataFrame([new_entry])

    corr_df = pd.concat([corr_df, new_df], ignore_index=True)
    corr_df.to_csv(f"{save_dir}/corr.csv", index=False)

    return test_data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--encoder", type=str)
    parser.add_argument("--model_type", choices=["all_atom", "fa", "decoy"], default="allatom")
    parser.add_argument("--ckpt", default="ckpts/model.skempi.allatom")
    parser.add_argument("--save_dir", default="ckpts/tmp")
    parser.add_argument("--noise_scaler", default="cosine")
    args = parser.parse_args()
    gpu, ckpt, model_type, save_dir = args.gpu, args.ckpt, args.model_type, args.save_dir
    eval_skempi(gpu, ckpt, model_type, save_dir)