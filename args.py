"""Common argparse builder.

Only flags actually consumed by the live training/evaluation path are
included. The following flags from the original scripts have been removed:

  Pure dead code:  --fp16, --fp16_opt_level, --lr_decay, --margin, --min_time,
                   --n_pair, --negative_weighting, --hard_negative_rate,
                   --use_mil, --sampled_use_mil
  GNN scaffolding: --gnn_model_type, --num_object, --node_feat_dim, --edge_dim,
                   --project_edge_dim, --no_skip, --last_average,
                   --no_beta_transformer, --node_features,
                   --data_geometric_path, --use_geometric_hdf5,
                   --fo_path, --stgraph_path
  Distillation:    --tradeoff_distill, --tradeoff_theta_2

None of these are read after the dead-code removal in this commit.
"""

import argparse


def build_parser(description, dataset_default):
    """Build a parser with all flags shared across the 4 entry scripts.

    `dataset_default` is "msrvtt" or "msvd"; it sets sensible defaults for the
    `--datatype` field. Dataset-specific defaults (data paths, lr, etc.) are
    set by the entry script after calling this.
    """
    p = argparse.ArgumentParser(description=description)

    p.add_argument("--do_train", action="store_true",
                   help="Run training (only used by train_*.py).")
    p.add_argument("--do_eval", action="store_true",
                   help="Run evaluation on the test split.")

    # Data
    p.add_argument("--data_path", type=str, required=True,
                   help="Caption JSON for the dataset.")
    p.add_argument("--features_path_cause", type=str, required=True,
                   help="Pickle of cause CLIP4Clip features.")
    p.add_argument("--features_path_effect", type=str, required=True,
                   help="Pickle of effect CLIP4Clip features.")
    p.add_argument("--datatype", default=dataset_default, type=str,
                   choices=["msrvtt", "msvd"],
                   help="Which dataset is being trained/evaluated.")

    # Optimisation
    p.add_argument("--num_thread_reader", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--coef_lr", type=float, default=0.1,
                   help="LR multiplier for the BERT branch.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--batch_size_val", type=int, default=64)
    p.add_argument("--warmup_proportion", type=float, default=0.1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_display", type=int, default=100)

    # Architecture
    p.add_argument("--max_words", type=int, default=48)
    p.add_argument("--max_frames", type=int, default=20)
    p.add_argument("--feature_framerate", type=int, default=1)
    p.add_argument("--video_dim", type=int, default=512)
    p.add_argument("--text_num_hidden_layers", type=int, default=12)
    p.add_argument("--visual_num_hidden_layers", type=int, default=2)
    p.add_argument("--decoder_num_hidden_layers", type=int, default=2)
    p.add_argument("--d_model", type=int, default=512)

    # Pretrained configs / checkpoints
    p.add_argument("--bert_model", default="modules/bert-model", type=str)
    p.add_argument("--visual_model", default="modules/visual-base", type=str)
    p.add_argument("--decoder_model", default="modules/decoder-base", type=str)
    p.add_argument("--init_model", default=None, type=str,
                   help="Optional initial checkpoint (e.g. UniVL or prior CEN run).")
    p.add_argument("--do_lower_case", action="store_true")
    p.add_argument("--cache_dir", default="", type=str)

    # Output / housekeeping
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--n_gpu", type=int, default=1)
    p.add_argument("--world_size", type=int, default=0)
    p.add_argument("--local_rank", type=int, default=0)

    # Early stopping
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--patience_metric", type=str, default="CIDEr")
    p.add_argument("--target_metric", type=str, default="CIDEr")

    return p


def finalise(args):
    """Validate and post-process parsed args."""
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1 (got %d)"
            % args.gradient_accumulation_steps
        )
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of --do_train or --do_eval must be set.")
    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)
    return args
