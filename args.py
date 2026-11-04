import argparse
from utils import ALPHABET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gt_D", action="store_true")
    parser.add_argument("--use_vae_regressor", action="store_true")
    parser.add_argument("--train_path", default="data/DIPS-Het/data.pkl")
    parser.add_argument("--test_path", default="data/skempi/skempi_all.pkl")
    parser.add_argument("--save_dir", default="ckpts/tmp")
    parser.add_argument("--embedding", default=None)
    parser.add_argument("--embedding_esm2", default=None)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use when a single GPU is available",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=1000000)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument(
        "--schedule_type",
        type=str,
        default="interval",
        choices=["interval", "step_blocks", "epoch_phase"],
        help='"interval": use coarse/fine intervals; "step_blocks": alternate blocks of n/m steps; "epoch_phase": coarse first half epochs, fine second half',
    )
    parser.add_argument("--coarse_interval", type=int, default=1)
    parser.add_argument("--fine_interval", type=int, default=5)
    parser.add_argument(
        "--block_n",
        type=int,
        default=100,
        help="n steps block size for step_blocks scheduling",
    )
    parser.add_argument(
        "--block_m",
        type=int,
        default=100,
        help="m steps block size for step_blocks scheduling",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--exp_name", type=str, default="docking_test")

    parser.add_argument("--load_model", default=None)
    parser.add_argument("--all_atom", action="store_true", default=False)
    parser.add_argument("--fa", action="store_true", default=False)
    parser.add_argument("--decoy", action="store_true", default=False)
    parser.add_argument("--gaussian", action="store_true", default=False)
    parser.add_argument("--nosidechain", action="store_true", default=False)
    parser.add_argument("--nolm", action="store_true", default=False)

    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--embedding_size", type=int, default=2560)
    parser.add_argument("--bert_size", type=int, default=2560)
    parser.add_argument("--patch_size", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--vocab_size", type=int, default=len(ALPHABET) * 3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=30.0)

    parser.add_argument("--resolution", type=str, default="backbone")
    parser.add_argument("--anneal_rate", type=float, default=0.95)
    parser.add_argument("--normalization", type=str, default="layer")
    parser.add_argument("--encoder", type=str, default="flash_ipa")
    parser.add_argument("--noise_scaler", type=str, default="cosine")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--require_pairwise_repr", action="store_true", default=False)
    parser.add_argument("--scalar_kv_dim", type=int, default=24)
    parser.add_argument("--point_kv_dim", type=int, default=6)
    parser.add_argument("--include_skempi", action="store_true", default=False)
    parser.add_argument("--ff_num_layers", type=int, default=2)
    parser.add_argument("--ff_mult", type=int, default=1)
    parser.add_argument("--initialization", type=str, default="xavier_u")
    parser.add_argument("--use_swiglu", action="store_true", default=False)
    parser.add_argument("--non_conservative", action="store_true", default=False)
    parser.add_argument(
        "--tr_s_min", default=0.1, help="min sigma for translational noise"
    )
    parser.add_argument(
        "--tr_s_max", default=20, help="max sigma for translational noise"
    )
    parser.add_argument(
        "--rot_s_min", default=0.1, help="min sigma for rotational noise"
    )
    parser.add_argument(
        "--rot_s_max", default=1.5, help="max sigma for rotational noise"
    )
    parser.add_argument("--ema_decay", default=0.999, type=float, help="EMA decay rate")
    parser.add_argument(
        "--gradient_accumulate_every",
        default=8,
        type=int,
        help="Accumulate gradients for this many steps",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="use tqdm output",
    )
    parser.add_argument("--ckpt", type=str, help="load from checkpoint")
    parser.add_argument(
        "--num_val_sample",
        type=int,
        default=30,
        help="number of entries from the validation set to sample for evaluation",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of conformations to sample before selecting the best one",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="If set with top_k and not non_conservative, select best conformation based on energy instead of CRMSD",
    )
    parser.add_argument(
        "--langevin_T", type=int, default=10, help="number of Langevin steps to run"
    )
    parser.add_argument("--n_supernodes", type=int, default=100)
    parser.add_argument("--noise_type", type=str, default="se3")
    parser.add_argument(
        "--noise_emb_dim", type=int, default=256, help="Dimension of noise embedding"
    )
    parser.add_argument(
        "--fourier_dim",
        type=int,
        default=8,
        help="Dimension of Fourier features for coordinates",
    )
    parser.add_argument(
        "--num_edge_feats",
        type=int,
        default=2,
        help="Number of Gaussians for distance embedding",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of training samples. Set to negative for full train set.",
    )
    parser.add_argument(
        "--test", action="store_true", help="If set, run on test set only"
    )

    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default=None,
        help="Path to VAE checkpoint for guided docking",
    )
    parser.add_argument(
        "--diff_ckpt",
        type=str,
        default=None,
        help="Path to diffusion checkpoint for guided docking",
    )
    parser.add_argument("--vae_lr", type=float, default=0.0001)
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Latent diffusion dimension"
    )
    parser.add_argument(
        "--protein_encoder_depth",
        type=int,
        default=5,
        help="Depth of protein encoder network",
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.001, help="Weight for KL divergence loss"
    )
    parser.add_argument("--vae_batch_size", type=int, default=16, help="VAE batch size")
    parser.add_argument(
        "--channel_depths",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Channel depths for VAE",
    )
    parser.add_argument(
        "--use_symmetry",
        action="store_true",
        help="Whether to use norms for distance symmetry or predict distances directly",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["distances", "contact"],
        default="distances",
        help="Target mode for VAE",
    )
    parser.add_argument(
        "--pred_type",
        type=str,
        choices=["direct", "ipa"],
        default="direct",
        help="Prediction type for the distance/contact map. 'direct' predicts distances/contacts directly, 'ipa' uses invariant point attention for coordinates.",
    )
    parser.add_argument(
        "--use_log_distances",
        action="store_true",
        help="Whether to use log-transformed distances",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Whether to run validation only",
    )
    parser.add_argument(
        "--use_guidedock",
        action="store_true",
        help="If set, use for sampling",
    )
    parser.add_argument(
        "--tr_grad_weight",
        type=float,
        default=0.25 * 0.1,
        help="Weight for translation gradient guidance",
    )
    parser.add_argument(
        "--rot_grad_weight",
        type=float,
        default=5.0,
        help="Weight for rotation gradient guidance",
    )
    parser.add_argument(
        "--save_trajectory",
        action="store_true",
        help="If set, save full docking trajectory",
    )
    parser.add_argument(
        "--num_D_samples",
        type=int,
        default=4,
        help="Number of distance matrix samples for guided docking",
    )
    parser.add_argument(
        "--cutoff_threshold",
        type=float,
        default=50.0,
        help="Cutoff threshold for distance guidance in angstroms",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=2,
        help="Number of parallel particles for guided docking",
    )
    parser.add_argument(
        "--save_meter",
        action="store_true",
        help="If set, save meter values during validation/testing",
    )

    return parser.parse_args()
