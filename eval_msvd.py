"""Evaluate a trained CEN checkpoint on the MSVD-CTN test split."""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.utils.data import DataLoader, SequentialSampler

import args as args_mod
import engine
from dataloaders.dataloader_msvd_caption_ce import MSVD_Caption_DataLoader
from modules.tokenization import BertTokenizer


def parse_args():
    p = args_mod.build_parser("Evaluate CEN on MSVD-CTN", dataset_default="msvd")
    p.set_defaults(do_eval=True, max_words=48, max_frames=20, batch_size_val=64)
    args = p.parse_args()
    args.do_eval = True
    return args_mod.finalise(args)


def dataloader_test(args, tokenizer):
    ds = MSVD_Caption_DataLoader(
        data_path=args.data_path,
        features_path_cause=args.features_path_cause,
        features_path_effect=args.features_path_effect,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="test",
        inference_path=None,
    )
    loader = DataLoader(
        ds, sampler=SequentialSampler(ds),
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader, pin_memory=False, drop_last=False,
    )
    return loader, len(ds)


def main():
    torch.distributed.init_process_group(backend="nccl")
    args = parse_args()
    args, logger = engine.set_seed_logger(args)
    device, _ = engine.init_device(args, logger)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.do_lower_case)
    model = engine.init_model(args, device)

    dataloaders = {"test": dataloader_test(args, tokenizer)}
    engine.run_evaluation(args, tokenizer, model, dataloaders, device, logger)


if __name__ == "__main__":
    main()
