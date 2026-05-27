"""Shared training and evaluation engine for CEN (Cause-Effect Network).

This module factors out everything that the four entry-point scripts
(`train_msrvtt.py`, `train_msvd.py`, `eval_msrvtt.py`, `eval_msvd.py`) used to
duplicate verbatim:

* random seeding and distributed device setup,
* model construction (`CaptionGenerator.from_pretrained`),
* optimiser construction (separate LR for BERT vs. non-BERT params),
* beam-search decoding helpers,
* `train_epoch` and `eval_epoch`,
* a single `score(...)` based on pycocoevalcap (BLEU-1..4 / ROUGE-L / CIDEr).

The original scripts kept two `score()` definitions, an unused `NLGEval`
object, a duplicated `dataloader_msvd_*` factory, and a commented-out
`theta_1`/GNN distillation path. All of that has been removed.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from tqdm import tqdm

from modules.beam import Beam
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_ce import CaptionGenerator
from modules.optimization import BertAdam
from utils import get_logger


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def set_seed_logger(args):
    """Seed RNGs, set CUDA device, create output dir, and return a logger."""
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< %s: %s", key, args.__dict__[key])

    return args, logger


def init_device(args, logger):
    """Pick CUDA device, validate per-GPU batch sizes."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu", args.local_rank
    )
    n_gpu = torch.distributed.get_world_size()
    logger.info("device: %s n_gpu: %d", device, n_gpu)
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "batch_size ({}) and batch_size_val ({}) must be divisible by n_gpu ({})".format(
                args.batch_size, args.batch_size_val, args.n_gpu
            )
        )
    return device, n_gpu


# ---------------------------------------------------------------------------
# Model + optimiser
# ---------------------------------------------------------------------------

def init_model(args, device):
    """Build CaptionGenerator from pretrained BERT/visual/decoder configs."""
    if args.init_model:
        state_dict = torch.load(args.init_model, map_location="cpu")
    else:
        state_dict = None
    cache_dir = args.cache_dir or os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed"
    )
    model = CaptionGenerator.from_pretrained(
        args.bert_model,
        args.visual_model,
        args.decoder_model,
        cache_dir=cache_dir,
        state_dict=state_dict,
        task_config=args,
    )
    model.to(device)
    return model


def prep_optimizer(args, model, num_train_optimization_steps, coef_lr=1.0):
    """BertAdam with separate LR for BERT vs. non-BERT weights, plus DDP wrap."""
    if hasattr(model, "module"):
        model = model.module

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    params = list(model.named_parameters())

    def in_bert(n):
        return "bert." in n

    def is_no_decay(n):
        return any(nd in n for nd in no_decay)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params if is_no_decay(n) and in_bert(n)],
            "weight_decay": 0.01,
            "lr": args.lr * coef_lr,
        },
        {
            "params": [p for n, p in params if is_no_decay(n) and not in_bert(n)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in params if not is_no_decay(n) and in_bert(n)],
            "weight_decay": 0.0,
            "lr": args.lr * coef_lr,
        },
        {
            "params": [p for n, p in params if not is_no_decay(n) and not in_bert(n)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.lr,
        warmup=args.warmup_proportion,
        schedule="warmup_linear",
        t_total=num_train_optimization_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
    )
    return optimizer, None, model  # scheduler is folded into BertAdam


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_model(epoch, args, model, logger, type_name=""):
    to_save = model.module if hasattr(model, "module") else model
    suffix = "" if type_name == "" else type_name + "."
    output_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format(suffix, epoch)
    )
    torch.save(to_save.state_dict(), output_file)
    logger.info("Model saved to %s", output_file)
    return output_file


def load_model(args, device, logger, model_file):
    if not model_file or not os.path.exists(model_file):
        return None
    state_dict = torch.load(model_file, map_location="cpu")
    if args.local_rank == 0:
        logger.info("Model loaded from %s", model_file)
    cache_dir = args.cache_dir or os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed"
    )
    model = CaptionGenerator.from_pretrained(
        args.bert_model,
        args.visual_model,
        args.decoder_model,
        cache_dir=cache_dir,
        state_dict=state_dict,
        task_config=args,
    )
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Metrics — pycocoevalcap (BLEU-1..4, ROUGE-L, CIDEr).
# METEOR / SPICE are intentionally omitted: they require Java and a separate
# download path that the live evaluation code never used correctly.
# ---------------------------------------------------------------------------

def score(refs, hyps):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        s, _ = scorer.compute_score(refs, hyps)
        if isinstance(method, list):
            for m, sm in zip(method, s):
                final_scores[m] = sm
        else:
            final_scores[method] = s
    return final_scores


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(epoch, args, model, train_dataloader, optimizer, scheduler,
                global_step, logger):
    torch.cuda.empty_cache()
    model.train()
    start = time.time()
    total_loss = 0.0
    log_step = args.n_display
    device = next(model.parameters()).device

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        (input_ids, input_mask, segment_ids,
         video_cause, video_effect, video_mask_cause, video_mask_effect,
         pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,
         pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids) = batch

        decoder_scores = model(
            video_cause, video_effect, video_mask_cause, video_mask_effect,
            input_caption_ids=pairs_input_caption_ids,
            decoder_mask=pairs_decoder_mask,
        )

        pairs_output_caption_ids = pairs_output_caption_ids.view(
            -1, pairs_output_caption_ids.shape[-1]
        )

        vocab_size = model.module.bert_config.vocab_size
        loss = model.module.decoder_loss_fct(
            decoder_scores.view(-1, vocab_size),
            pairs_output_caption_ids.view(-1),
        )

        if args.n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        total_loss += float(loss)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scheduler is not None:
                scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and args.local_rank == 0:
                lrs = "-".join(
                    str("%.6f" % itm) for itm in sorted(set(optimizer.get_lr()))
                )
                logger.info(
                    "Epoch: %d/%d, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f",
                    epoch + 1, args.epochs, step + 1, len(train_dataloader),
                    lrs, float(loss),
                    (time.time() - start) / (log_step * args.gradient_accumulation_steps),
                )
                start = time.time()

    return total_loss / max(1, len(train_dataloader)), global_step


# ---------------------------------------------------------------------------
# Beam search helpers
# ---------------------------------------------------------------------------

def _inst_idx_to_position(inst_idx_list):
    return {inst_idx: pos for pos, inst_idx in enumerate(inst_idx_list)}


def _collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    _, *d_hs = beamed_tensor.size()
    n_curr = len(curr_active_inst_idx)
    new_shape = (n_curr * n_bm, *d_hs)
    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    return beamed_tensor.view(*new_shape)


def _collate_active_info(input_tuples, inst_idx_to_position_map,
                         active_inst_idx_list, n_bm, device):
    visual_output_rpt, video_mask_rpt = input_tuples
    n_prev = len(inst_idx_to_position_map)
    active_inst_idx = torch.LongTensor(
        [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    ).to(device)
    return (
        (
            _collect_active_part(visual_output_rpt, active_inst_idx, n_prev, n_bm),
            _collect_active_part(video_mask_rpt, active_inst_idx, n_prev, n_bm),
        ),
        _inst_idx_to_position(active_inst_idx_list),
    )


def _beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                      inst_idx_to_position_map, n_bm, device, input_tuples):
    visual_output_rpt, video_mask_rpt = input_tuples

    dec_partial_seq = torch.stack(
        [b.get_current_state() for b in inst_dec_beams if not b.done]
    ).to(device).view(-1, len_dec_seq)
    next_decoder_mask = torch.ones(dec_partial_seq.size(), dtype=torch.uint8).to(device)

    dec_output = decoder(
        visual_output_rpt, video_mask_rpt,
        dec_partial_seq, next_decoder_mask,
        shaped=True, get_logits=True,
    )
    dec_output = dec_output[:, -1, :]
    word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
    n_active = len(inst_idx_to_position_map)
    word_prob = word_prob.view(n_active, n_bm, -1)

    active = []
    for inst_idx, inst_pos in inst_idx_to_position_map.items():
        if not inst_dec_beams[inst_idx].advance(word_prob[inst_pos]):
            active.append(inst_idx)
    return active


def _collect_hypotheses(inst_dec_beams, n_best=1):
    all_hyp = []
    for inst in inst_dec_beams:
        _, tail_idxs = inst.sort_scores()
        all_hyp.append([inst.get_hypothesis(i) for i in tail_idxs[:n_best]])
    return all_hyp


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _detokenise(token_ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    for special in ("[SEP]", "[PAD]"):
        if special in tokens:
            tokens = tokens[: tokens.index(special)]
    text = " ".join(tokens).replace(" ##", "").strip("##").strip()
    return text


def eval_epoch(args, model, test_dataloader, tokenizer, device, logger,
               n_bm=5):
    if hasattr(model, "module"):
        model = model.module.to(device)
    model.eval()

    all_hyps, all_refs_per_sample = [], []
    for batch in tqdm(test_dataloader, desc="validation"):
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        (_input_ids, _input_mask, _segment_ids,
         video_cause, video_effect, video_mask_cause, video_mask_effect,
         _pm_text, _pt_labels, _masked_video, _v_labels_idx,
         _pic_ids, _pd_mask, pairs_output_caption_ids) = batch

        with torch.no_grad():
            v_cause = model.get_visual_output(model.visual1, video_cause, video_mask_cause)
            v_effect = model.get_visual_output(model.visual2, video_effect, video_mask_effect)

            visual_output = torch.cat((v_cause, v_effect), dim=1)
            video_mask = torch.cat((video_mask_cause, video_mask_effect), dim=1)
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            n_inst, len_v, v_h = visual_output.size()
            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(
                n_inst * n_bm, len_v, v_h
            )
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            inst_dec_beams = [
                Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)
            ]
            active = list(range(n_inst))
            pos_map = _inst_idx_to_position(active)

            for L in range(1, args.max_words + 1):
                active = _beam_decode_step(
                    model.decoder_caption, inst_dec_beams, L, pos_map,
                    n_bm, device, (visual_output_rpt, video_mask_rpt),
                )
                if not active:
                    break
                (visual_output_rpt, video_mask_rpt), pos_map = _collate_active_info(
                    (visual_output_rpt, video_mask_rpt), pos_map, active, n_bm, device,
                )

            hyps = _collect_hypotheses(inst_dec_beams, n_best=1)
            for i in range(n_inst):
                all_hyps.append(_detokenise(hyps[i][0], tokenizer))

            captions = pairs_output_caption_ids.view(
                -1, pairs_output_caption_ids.shape[-1]
            ).cpu().numpy()
            for c in captions:
                all_refs_per_sample.append(_detokenise(c, tokenizer))

    # Persist raw outputs for inspection
    with open(os.path.join(args.output_dir, "hyp.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_hyps) + "\n")
    with open(os.path.join(args.output_dir, "ref.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_refs_per_sample) + "\n")

    # Build refs from dataset metadata so each video's full reference set is used
    sentences_dict = test_dataloader.dataset.sentences_dict
    video_sentences_dict = test_dataloader.dataset.video_sentences_dict
    all_refs = []
    for idx in range(len(sentences_dict)):
        video_id, _ = sentences_dict[idx]
        all_refs.append(video_sentences_dict[video_id])

    hyp_dict = {i: [all_hyps[i]] for i in range(len(all_hyps))}
    ref_dict = {i: all_refs[i] for i in range(len(all_refs))}

    metrics = score(ref_dict, hyp_dict)
    logger.info(
        ">>> BLEU_1: %.4f, BLEU_2: %.4f, BLEU_3: %.4f, BLEU_4: %.4f",
        metrics["Bleu_1"], metrics["Bleu_2"], metrics["Bleu_3"], metrics["Bleu_4"],
    )
    logger.info(
        ">>> ROUGE_L: %.4f, CIDEr: %.4f",
        metrics["ROUGE_L"], metrics["CIDEr"],
    )
    return metrics


# ---------------------------------------------------------------------------
# High-level orchestration: training and evaluation drivers
# ---------------------------------------------------------------------------

def run_training(args, tokenizer, model, dataloaders, device, logger):
    """Standard train-with-early-stopping loop driven from the entry scripts.

    `dataloaders` is a dict with keys 'train', 'val', 'test' returning
    (loader, length[, sampler]) — sampler only required for 'train'.
    """
    train_dataloader, train_length, train_sampler = dataloaders["train"]
    val_dataloader, val_length = dataloaders["val"]
    test_dataloader, test_length = dataloaders["test"]

    num_train_optimization_steps = (
        int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
        / args.gradient_accumulation_steps
    ) * args.epochs

    coef_lr = 1.0 if args.init_model else args.coef_lr
    optimizer, scheduler, model = prep_optimizer(
        args, model, num_train_optimization_steps, coef_lr=coef_lr
    )

    if args.local_rank == 0:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d",
                    num_train_optimization_steps * args.gradient_accumulation_steps)

    best_score = {"CIDEr": 1e-5}
    best_file = {"CIDEr": None}
    assert args.target_metric in best_score
    assert args.patience_metric in best_score

    global_step = 0
    stop_signal = torch.zeros(2).cuda()  # [0]=patience counter, [1]=metric collapse

    # Sanity-check eval on test before training (matches original behaviour)
    eval_epoch(args, model, test_dataloader, tokenizer, device, logger)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        tr_loss, global_step = train_epoch(
            epoch, args, model, train_dataloader, optimizer, scheduler,
            global_step, logger,
        )
        logger.info("Epoch %d/%d finished, train loss: %f",
                    epoch + 1, args.epochs, tr_loss)
        output_file = save_model(epoch, args, model, logger)

        if args.local_rank == 0:
            if epoch > 0:
                metrics = eval_epoch(args, model, val_dataloader,
                                     tokenizer, device, logger)
                m = args.target_metric
                if metrics[m] <= 0.001:
                    logger.warning("Metric collapse on %s; stopping.", m)
                    stop_signal[1] = 1
                elif best_score[m] <= metrics[m]:
                    best_score[m] = metrics[m]
                    best_file[m] = output_file
                    if m == args.patience_metric:
                        stop_signal[0] = 0
                else:
                    if m == args.patience_metric:
                        stop_signal[0] += 1
                logger.info("Best %s so far: %.4f (%s)", m, best_score[m], best_file[m])

                for gp in range(1, args.n_gpu):
                    torch.distributed.send(stop_signal, dst=gp)

                if stop_signal[0] >= args.patience:
                    logger.warning("Early stopping at epoch %d", epoch + 1)
                    break
                if stop_signal[1] == 1:
                    break
            else:
                for gp in range(1, args.n_gpu):
                    torch.distributed.send(stop_signal, dst=gp)
                logger.warning("Skipping validation after epoch %d", epoch + 1)
        else:
            torch.distributed.recv(stop_signal, src=0)
            if stop_signal[0] >= args.patience or stop_signal[1] == 1:
                break

    if args.local_rank == 0:
        m = args.target_metric
        best = load_model(args, device, logger, best_file[m])
        if best is not None:
            metrics = eval_epoch(args, best, test_dataloader,
                                 tokenizer, device, logger)
            logger.info("Test metrics for best %s model (%s): %s",
                        m, best_file[m], metrics)


def run_evaluation(args, tokenizer, model, dataloaders, device, logger):
    """Single evaluation pass on the test split."""
    test_dataloader, test_length = dataloaders["test"]
    if args.local_rank == 0:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
    return eval_epoch(args, model, test_dataloader, tokenizer, device, logger)
