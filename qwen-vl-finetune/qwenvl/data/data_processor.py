import json
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    # (Logging code omitted for brevity, keeping original logic...)
    
    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        if turn["from"] == "system":
            continue  # Skip system messages, as we add them separately
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def get_system_prompt(task_type: str) -> str:
    """
    Determine system prompt based on the task type injected during dataset loading.
    """
    if task_type == "short_answer":
        return "Answer the question using a single word or phrase."
    elif task_type == "multiple_choice":
        return "Choose the correct option from the given choices."
    elif task_type == "grounding":
        return "Provide the bounding box coordinates of the object described."
    else:
        # Default / Chat
        return "You are a helpful assistant."


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    # --- MODIFIED: Use injected task_type ---
    task_type = source.get("task_type", "chat")
    system_prompt = get_system_prompt(task_type)
    
    messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        # 77091 is the token ID for 'im_start' or similar delimiter in Qwen
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            # 151645 is likely 'im_end'
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        
        # ... (Video params init kept same) ...
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            
            # --- MODIFIED: Determine Task Type for this specific dataset ---
            # We determine the type based on the dataset name/path here ONCE,
            # instead of guessing per-image later.
            ds_name = (data.get("annotation_path", "") + data.get("data_path", "")).lower()
            
            if any(x in ds_name for x in ["ai2d", "scienceqa", "mmmu", "aokvqa"]):
                task_type = "multiple_choice"
            elif any(x in ds_name for x in ["docvqa", "dvqa", "ocrvqa", "chartqa", "textvqa", "vqav2", "gqa"]):
                task_type = "short_answer"
            elif any(x in ds_name for x in ["refcoco", "visual_genome", "vg"]):
                task_type = "grounding"
            else:
                task_type = "chat" # Default for LLaVA, ShareGPT4V
            
            rank0_print(f"Dataset: {data.get('annotation_path')} -> Assigned Task Type: {task_type}")

            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            
            for ann in annotations:
                # Inject the task type into every sample
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                        sub_ann["task_type"] = task_type
                else:
                    ann["data_path"] = data["data_path"]
                    ann["task_type"] = task_type
                    
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        # ... (Rest of init kept same) ...
        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    # ... (Rest of class methods __len__, __getitem__, etc. kept same) ...
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list
    
    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        # ... (Get item logic same) ...
        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                pass

        # Final attempt
        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        # This calls preprocess_qwen_visual which now handles task_type
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
        )
        # ... (Rest of _get_item kept exactly the same) ...
        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        
        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:
        # Wrapper for packing, mostly relies on _get_item
        if isinstance(sources, dict):
            if isinstance(sources, dict): # Check original code var names
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert len(source) == 1
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [d["pixel_values"] for d in data_list if "pixel_values" in d], dim=0
                        ),
                        "image_grid_thw": torch.cat(
                            [d["image_grid_thw"] for d in data_list if "image_grid_thw" in d], dim=0
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [d["pixel_values_videos"] for d in data_list if "pixel_values_videos" in d], dim=0
                        ),
                        "video_grid_thw": torch.cat(
                            [d["video_grid_thw"] for d in data_list if "video_grid_thw" in d], dim=0
                        ),
                    }
                )
            return new_data_dict

# ... (DataCollator classes pad_and_cat kept same) ...
def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)
    stacked_tensor = torch.cat(padded_tensors, dim=1)
    return stacked_tensor

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # (Standard collator code omitted for brevity but assumes same as input)
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        
        if len(images) != 0:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat([inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst], dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        if len(videos) != 0:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat([inst["video_grid_thw"] for inst in instances if "video_grid_thw" in inst], dim=0)
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None
            
        batch["position_ids"] = position_ids
        return batch

@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # (Same as input)
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(itertools.chain(*(instance["attention_mask"] for instance in instances if "attention_mask" in instance)))
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        # (Flattened media handling same as input)
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        if len(images) != 0:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat([inst["image_grid_thw"] for inst in instances if "image_grid_thw" in inst], dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None
        if len(videos) != 0:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat([inst["video_grid_thw"] for inst in instances if "video_grid_thw" in inst], dim=0)
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        return batch

def make_supervised_data_module(processor, data_args) -> Dict:
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    eval_dataset = None
    if data_args.eval_dataset_use:
        # Create eval data_args with eval_dataset_use
        eval_data_args = data_args.__class__(
            dataset_use=data_args.eval_dataset_use,
            eval_dataset_use="",  # not needed for eval
            model_type=data_args.model_type,
            data_flatten=data_args.data_flatten,
            data_packing=data_args.data_packing,
            base_interval=data_args.base_interval,
            max_pixels=data_args.max_pixels,
            min_pixels=data_args.min_pixels,
            video_max_frames=data_args.video_max_frames,
            video_min_frames=data_args.video_min_frames,
            video_max_pixels=data_args.video_max_pixels,
            video_min_pixels=data_args.video_min_pixels,
            video_fps=data_args.video_fps,
        )
        eval_dataset = LazySupervisedDataset(processor, data_args=eval_data_args)
    
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

if __name__ == "__main__":
    pass