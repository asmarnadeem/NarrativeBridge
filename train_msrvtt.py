"""Train CEN on the MSRVTT-CTN dataset."""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.utils.data import DataLoader, SequentialSampler

import args as args_mod
import engine
from dataloaders.dataloader_msrvtt_caption_ce import MSRVTT_Caption_DataLoader
from modules.tokenization import BertTokenizer


def parse_args():
    p = args_mod.build_parser("Train CEN on MSRVTT-CTN", dataset_default="msrvtt")
    # MSRVTT default overrides matching the paper
    p.set_defaults(lr=5e-5, batch_size=64, batch_size_val=64,
                   max_words=48, max_frames=20)
    return args_mod.finalise(p.parse_args())


def dataloader_train(args, tokenizer):
    ds = MSRVTT_Caption_DataLoader(
        json_path=args.data_path,
        features_path_cause=args.features_path_cause,
        features_path_effect=args.features_path_effect,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        inference_path=None,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    loader = DataLoader(
        ds, batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader, pin_memory=False,
        shuffle=False, sampler=sampler, drop_last=True,
    )
    return loader, len(ds), sampler


def dataloader_val_test(args, tokenizer, split_type):
    ds = MSRVTT_Caption_DataLoader(
        json_path=args.data_path,
        features_path_cause=args.features_path_cause,
        features_path_effect=args.features_path_effect,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
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

    dataloaders = {
        "train": dataloader_train(args, tokenizer),
        "val":   dataloader_val_test(args, tokenizer, "val"),
        "test":  dataloader_val_test(args, tokenizer, "test"),
    }

    engine.run_training(args, tokenizer, model, dataloaders, device, logger)


if __name__ == "__main__":
    main()
