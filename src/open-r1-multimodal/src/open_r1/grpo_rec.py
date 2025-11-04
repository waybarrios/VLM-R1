# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import pickle
import hashlib

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
import numpy as np
import torch
from datasets import load_from_disk

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    use_huggingface_dataset: bool = field(
        default=False,
        metadata={"help": "Whether to use HuggingFace dataset format instead of JSON/JSONL"},
    )
    task_type: str = field(
        default="rec",
        metadata={"help": "Task type for the VLM module. Possible values: 'rec', 'vqa', 'ic', 'odLength'"},
    )
    use_llm_judge: bool = field(
        default=False,
        metadata={"help": "Whether to use LLM-based judge for accuracy evaluation (requires Ollama)"},
    )
    llm_judge_model: str = field(
        default="gpt-oss:20b",
        metadata={"help": "Ollama model name for LLM judge (e.g., 'gpt-oss:20b', 'llama3.2', 'mistral', 'phi3')"},
    )
    llm_judge_base_url: str = field(
        default="http://localhost:11434/v1",
        metadata={"help": "Base URL for Ollama API (OpenAI-compatible endpoint)"},
    )
    shuffle_train_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str, seed: Optional[int] = None):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.seed = seed
        self.list_data_dict = []
        self.question_template = question_template
        self.use_huggingface = script_args.use_huggingface_dataset

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"Random seed set to: {seed}")

        # Create cache filename based on data_path and seed
        cache_key = f"{data_path}_{seed}_{script_args.shuffle_train_dataset}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = os.path.join(os.path.dirname(data_path), ".dataset_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"dataset_cache_{cache_hash}.pkl")

        # Try to load from cache
        if os.path.exists(self.cache_file):
            print(f"📦 Loading dataset from cache: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    self.list_data_dict = pickle.load(f)
                print(f"✓ Loaded {len(self.list_data_dict)} samples from cache!")
                return  # Skip loading and processing
            except Exception as e:
                print(f"⚠ Cache loading failed: {e}. Loading from source...")

        if self.use_huggingface:
            # Load HuggingFace dataset from disk
            print(f"Loading HuggingFace dataset from: {data_path}")
            print("This may take a while if dataset is large or on network storage...")
            dataset = load_from_disk(data_path)
            print(f"✓ Dataset loaded! Total rows: {len(dataset)}")

            # Convert HuggingFace dataset to list of dicts
            print("Converting dataset to internal format...")
            dataset_len = len(dataset)
            print_every = max(1, dataset_len // 10)  # Print progress every 10%

            for idx, item in enumerate(dataset):
                self.list_data_dict.append({
                    'image': item['image'],  # PIL Image
                    'question': item['question'],
                    'answer': item['answer'],
                    'reference_steps': item.get('reference_steps', []),
                    'choices': item.get('choices', []),
                    'options': item.get('options', []),
                    'source': item.get('source', ''),
                })
                if (idx + 1) % print_every == 0:
                    progress = (idx + 1) / dataset_len * 100
                    print(f"  Progress: {idx + 1}/{dataset_len} ({progress:.1f}%)")

            print(f"✓ Loaded {len(self.list_data_dict)} samples from HuggingFace dataset at {data_path}")
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        # Shuffle the entire dataset if requested
        if script_args.shuffle_train_dataset:
            random.shuffle(self.list_data_dict)
            print(f"Dataset shuffled with seed {self.seed}")

        print(f"Total samples in dataset: {len(self.list_data_dict)}")

        # Save to cache for future runs
        try:
            print(f"💾 Saving dataset to cache: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.list_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✓ Cache saved! Future runs will load instantly.")
        except Exception as e:
            print(f"⚠ Failed to save cache (non-critical): {e}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        example = self.list_data_dict[i]

        if self.use_huggingface:
            # HuggingFace dataset format
            # Format the user question with choices/options
            if example.get("choices"):
                formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(example["choices"])])
            elif example.get("options"):
                formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(example["options"])])
            else:
                formatted_options = ""

            user_question = f"{example['question']}\n\n{formatted_options}".strip()
            user_instruction = self.question_template.replace("{USER_INSTRUCTION}", user_question)

            # Image is already PIL Image from HuggingFace dataset
            image = example['image']

            # Create prompt without system message
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_instruction},
                    ],
                },
            ]

            return {
                'image': image,
                'problem': user_question,
                'solution': example['answer'],
                'reference_steps': example.get('reference_steps', []),
                'prompt': prompt,
            }
        else:
            # Original JSON/JSONL format
            def make_conversation(example):
                return {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": example["problem"]},
                    ],
                }
            QUESTION_TEMPLATE = self.question_template
            def make_conversation_image(example):
                return {
                    "prompt": [
                        # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                            ],
                        },
                    ],
                }

            image_root = self.script_args.image_root
            if 'image' in example:
                image_path = os.path.join(image_root, example['image'])
                # In case the image is not found
                while not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found, randomly selecting another image")
                    new_index = random.randint(0, len(self.list_data_dict)-1)
                    example = self.list_data_dict[new_index]
                    image_path = os.path.join(image_root, example['image'])
                image = Image.open(image_path).convert("RGB")
            else:
                image = None


            return {
                'image': image,
                'problem': example['problem'],
                'solution': example['solution'],
                'prompt': make_conversation_image(example)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
            }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    elif "glm" in model_name_or_path.lower():
        return GLMVModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Configure LLM judge if using VQA task
    if script_args.task_type == "vqa":
        vlm_module_cls.configure_llm_judge(
            use_llm_judge=script_args.use_llm_judge,
            llm_judge_model=script_args.llm_judge_model,
            llm_judge_base_url=script_args.llm_judge_base_url
        )
        print(f"LLM Judge configured: use_llm_judge={script_args.use_llm_judge}, model={script_args.llm_judge_model}")

    # Load the reward functions based on task type
    reward_funcs = [vlm_module_cls.select_reward_func(func, script_args.task_type) for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    print("task_type:", script_args.task_type)

    # Load the dataset
    dataset = LazySupervisedDataset(
        script_args.dataset_name,
        script_args,
        question_template=vlm_module_cls.get_question_template(task_type=script_args.task_type),
        seed=training_args.seed
    )

    trainer_cls = VLMGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
