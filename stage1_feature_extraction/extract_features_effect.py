import os
import sys
import glob
import pickle
import pathlib

import torch
from tqdm import tqdm

parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'dataloaders'))

try:
    from modules.modeling import CLIP4Clip
    from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
except ImportError:
    PYTORCH_PRETRAINED_BERT_CACHE = '.cache'

device = torch.device('cuda')

# ── Configuration ─────────────────────────────────────────────────────────────
class args:
    msvd = False            # set True to extract MSVD features
    dset = '../'            # path to repository root
    max_frames = 20
    max_words = 48
    feature_framerate = 1
    eval_frame_order = 0
    slice_framepos = 2
    cross_model = "cross-base"
    cache_dir = ''
    local_rank = 0

# Path to your trained CLIP4Clip effect model checkpoint
# Train CLIP4Clip on effect captions: https://github.com/ArrowLuo/CLIP4Clip
EFFECT_MODEL_PATH = "/path/to/clip4clip_effect/pytorch_model.bin"
BATCH_SAVE_SIZE = 50
# ─────────────────────────────────────────────────────────────────────────────

if args.msvd:
    dset_path = os.path.join(args.dset, 'dataset', 'MSVD')
    args.features_path = os.path.join(dset_path, 'raw')
    args.data_path = os.path.join(dset_path, 'captions', 'youtube_mapping.txt')
    args.max_words = 30
    save_dir = "extracted/msvd"
    save_file = save_dir + '/MSVD_effect_features.pickle'
    from dataloader_msvd import MSVD_Loader
    videos = MSVD_Loader(
        data_path=args.data_path, features_path=args.features_path,
        max_words=args.max_words, feature_framerate=args.feature_framerate,
        max_frames=args.max_frames, frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos, transform_type=0,
    )
else:
    dset_path = os.path.join(args.dset, 'dataset', 'MSRVTT')
    args.features_path = os.path.join(dset_path, 'raw')
    args.data_path = os.path.join(dset_path, 'MSRVTT_mistral_data_causal.json')
    args.msrvtt_csv = os.path.join(dset_path, 'msrvtt.csv')
    args.max_words = 73
    save_dir = "extracted/msrvtt"
    save_file = save_dir + '/MSRVTT_effect_features.pickle'
    from dataloader_msrvtt import MSRVTT_RawDataLoader
    videos = MSRVTT_RawDataLoader(
        csv_path=args.msrvtt_csv, features_path=args.features_path,
        max_words=args.max_words, feature_framerate=args.feature_framerate,
        max_frames=args.max_frames, frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos, transform_type=0,
    )

pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

model_state_dict = torch.load(EFFECT_MODEL_PATH, map_location='cpu')
cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
clip = model.clip.to(device)
clip.eval()

temp_dir = save_dir + '/temp_chunks_effect'
pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)

with torch.no_grad():
    data = {}
    chunk_counter = 0
    for i in tqdm(range(len(videos)), desc="Extracting effect features"):
        video_id, video, video_mask = videos[i]
        tensor = video[0]
        tensor = tensor[video_mask[0] == 1, :]
        tensor = torch.as_tensor(tensor).float()
        video_frame, num, channel, h, w = tensor.shape
        tensor = tensor.view(video_frame * num, channel, h, w)
        video_frame, channel, h, w = tensor.shape
        output = clip.encode_image(tensor.to(device), video_frame=video_frame,
                                   return_spatial=True).float().to(device)
        data[video_id] = output.detach().cpu().numpy()
        del output
        torch.cuda.empty_cache()
        if (i + 1) % BATCH_SAVE_SIZE == 0 or (i + 1) == len(videos):
            chunk_file = os.path.join(temp_dir, f'chunk_{chunk_counter}.pickle')
            with open(chunk_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            data = {}
            chunk_counter += 1

final_data = {}
for chunk_idx in tqdm(range(chunk_counter), desc="Merging chunks"):
    chunk_file = os.path.join(temp_dir, f'chunk_{chunk_idx}.pickle')
    with open(chunk_file, 'rb') as f:
        final_data.update(pickle.load(f))
    os.remove(chunk_file)

with open(save_file, 'wb') as f:
    pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

try:
    os.rmdir(temp_dir)
except OSError:
    pass

print(f"Saved {len(final_data)} videos to {save_file}")
