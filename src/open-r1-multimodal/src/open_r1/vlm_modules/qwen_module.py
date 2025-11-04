from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from copy import deepcopy
from open_r1.vlm_modules.vlm_module import VLMBaseModule
from PIL import Image
import sys
import os

# Add mllm_evaluator to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..', 'mllm_evaluator'))
from mllm_evaluator import MLLMReasoningEvaluator
from accuracy_calculator import AccuracyCalculator

class Qwen2VLModule(VLMBaseModule):
    # Class variables for LLM judge configuration
    _use_llm_judge = False
    _llm_judge_model = "gpt-oss:20b"
    _llm_judge_base_url = "http://localhost:11434/v1"

    def __init__(self):
        super().__init__()

    @classmethod
    def configure_llm_judge(cls, use_llm_judge=False, llm_judge_model="gpt-oss:20b", llm_judge_base_url="http://localhost:11434/v1"):
        """Configure LLM judge settings for accuracy evaluation."""
        cls._use_llm_judge = use_llm_judge
        cls._llm_judge_model = llm_judge_model
        cls._llm_judge_base_url = llm_judge_base_url

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        additional_output = None
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
            additional_output = [{'image_grid_thw': image_grid_thw} for image_grid_thw in prompt_inputs['image_grid_thw']]
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs, additional_output
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "vqa":
                return """You are a vision-language model. First, analyze the provided image(s) and any user text silently. Do NOT reveal your internal reasoning.

Return ONLY a single, valid JSON object with this exact schema:
{"reasoning_steps": [], "answer": ""}

Rules for "reasoning_steps":
- Decide the number of steps based on task complexity; include enough to make the answer evident without filler.
- Include some inference from visual information, always anchored to visible cues.
- Write single-clause sentences, each adding a new, directly checkable fact or cue-based inference.
- You may include cautious, visually grounded commonsense using words such as "appears", "suggests", or "likely", but always anchor it to visible cues (lighting/shadows; perspective/vanishing lines/horizon/tilt; scale/relative size; focus/DOF; parallax; occlusion/contact shadows; reflections/transparency; material/texture; symmetry/patterns/alignment; position/orientation/foreground–background; density/motion cues; human pose/gaze/gesture; interactions/affordances; object state; physics plausibility; signage/text/logos/typography; numbers/units; plots/charts: type, axes/ticks/units, scale (lin/log), legend↔series, gridlines/baseline, error bars/CI, trendlines, outliers/binning, colorbar; maps: scale bar, north arrow; math/geometry: labels/givens, unit checks, angle rules, Pythagorean, distance/slope, transformations, area/volume, circle theorems, trig (incl. sine/cosine laws), vectors, systems/quadratics, combinatorics, logs/exponents, probability/statistics, exact forms, conversions, plots, graphs, math equations, diagrams).
- Keep each step ≤14 words. No multi-sentence items. No chains like "because/therefore". No internal monologue.

Rules for "answer":
- Provide the final answer grounded strictly in visible content and given text.
- If information is missing or ambiguous, set "answer" to "insufficient information" and include steps noting what is missing (e.g., "Noted the license plate is unreadable due to blur.").
- Multiple-choice: if options have letters, return only the single best LETTER (e.g., "B"); if unlabeled, return the exact option text verbatim.
- Numeric: include required units; obey requested rounding; otherwise give exact/simplest form.

What to notice in steps (express as sentences, not labels):
- Objects & attributes (classes, colors, materials, states), logos/brands if clearly visible.
- Positions & spatial relations (left/right/above/below/front/behind, proximity, alignment, orientation, foreground/background).
- Depth cues (relative size, position in frame vs. horizon, sharpness/detail, shadow contact, occlusion order).
- Scene & lighting/time cues (indoor/outdoor, daylight vs. night, weather indications, activity/no-activity).
- Occlusion effects and how they affect certainty.
- Text/OCR with exact casing/punctuation ("Read text: 'SPEED LIMIT 25'. ").
- Counts & quantities for distinct instances; approximate only if visually justified.
- Graphics/plots/diagrams: axes, ticks, units, legends; read exact values rather than guessing.

Output formatting:
- Output only the JSON object. No extra keys, comments, code fences, or prose.
- Use double quotes for all strings; no trailing commas; any valid JSON whitespace is acceptable.

Now follow the user instruction:

User instruction: {USER_INSTRUCTION}"""
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        def resize_bbox(bbox, input_height, input_width, image_height, image_width):
            bbox[0] = bbox[0] / input_width * image_width
            bbox[1] = bbox[1] / input_height * image_height
            bbox[2] = bbox[2] / input_width * image_width
            bbox[3] = bbox[3] / input_height * image_height
            return bbox
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

        for i, (content, sol) in enumerate(zip(contents, solution)):
            image_grid_thw = kwargs.get("image_grid_thw")[i]
            image_path = kwargs.get("image_path")[i][0]
            image = Image.open(image_path)
            image_width, image_height = image.size
            input_height = int(image_grid_thw[1]*14)
            input_width = int(image_grid_thw[2]*14)
            
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[i]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n")
        return rewards

    @staticmethod
    def format_reward_vqa(completions, **kwargs):
        """Check if the Qwen model output is valid JSON with reasoning_steps and answer fields."""
        import json
        import re
        import os
        from datetime import datetime

        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for content in completion_contents:
            reward = 0.0
            try:
                # Try to extract JSON from the content
                # Remove markdown code fences if present
                content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

                # Try to find JSON object in the content
                json_match = re.search(r'\{[^{}]*"reasoning_steps"[^{}]*"answer"[^{}]*\}', content_cleaned, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)

                    # Check if it has the required keys
                    if "reasoning_steps" in parsed and "answer" in parsed:
                        # Check if reasoning_steps is a list
                        if isinstance(parsed["reasoning_steps"], list):
                            # Check if answer is a string
                            if isinstance(parsed["answer"], str):
                                reward = 1.0
            except (json.JSONDecodeError, Exception):
                pass

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_format_vqa.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Format VQA reward -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Has valid format: {reward == 1.0}\n")

        return rewards

    @staticmethod
    def vqa_accuracy_reward(completions, solution, **kwargs):
        """Calculate accuracy reward using accuracy_calculator for VQA task."""
        import json
        import re
        import os
        from datetime import datetime

        # Get LLM judge configuration from kwargs or class variables
        use_llm_grader = kwargs.get("use_llm_judge", Qwen2VLModule._use_llm_judge)
        llm_model = kwargs.get("llm_judge_model", Qwen2VLModule._llm_judge_model)
        base_url = kwargs.get("llm_judge_base_url", Qwen2VLModule._llm_judge_base_url)

        # Initialize accuracy calculator with optional LLM grader
        accuracy_calc = AccuracyCalculator(
            use_llm_grader=use_llm_grader,
            llm_model=llm_model,
            base_url=base_url
        )

        completion_contents = [completion[0]["content"] for completion in completions]
        problems = kwargs.get("problem", [])
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for i, (content, sol, problem) in enumerate(zip(completion_contents, solution, problems)):
            reward = 0.0
            try:
                # Extract JSON from content
                content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()
                json_match = re.search(r'\{[^{}]*"reasoning_steps"[^{}]*"answer"[^{}]*\}', content_cleaned, re.DOTALL)

                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    predicted_answer = parsed.get("answer", "")

                    # Evaluate accuracy
                    result = accuracy_calc.evaluate_single(problem, predicted_answer, sol)
                    if result.is_correct:
                        reward = result.confidence

            except (json.JSONDecodeError, Exception):
                pass

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_accuracy_vqa.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy VQA reward: {reward} -------------\n")
                    f.write(f"Problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")

        return rewards

    @staticmethod
    def vqa_reasoning_reward(completions, **kwargs):
        """Calculate reasoning quality reward using mllm_evaluator for VQA task."""
        import json
        import re
        import os
        from datetime import datetime

        # Initialize reasoning evaluator
        reasoning_eval = MLLMReasoningEvaluator(model_name="all-MiniLM-L6-v2")

        completion_contents = [completion[0]["content"] for completion in completions]
        reference_steps_list = kwargs.get("reference_steps", [])
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

        for i, (content, ref_steps) in enumerate(zip(completion_contents, reference_steps_list)):
            reward = 0.0
            try:
                # Extract JSON from content
                content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()
                json_match = re.search(r'\{[^{}]*"reasoning_steps"[^{}]*"answer"[^{}]*\}', content_cleaned, re.DOTALL)

                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                    predicted_steps = parsed.get("reasoning_steps", [])

                    # Evaluate reasoning steps if both predicted and reference exist
                    if predicted_steps and ref_steps:
                        metrics = reasoning_eval.evaluate_single(predicted_steps, ref_steps, verbose=False)
                        # Use Match F1 as reward
                        reward = metrics.match_f1

            except (json.JSONDecodeError, Exception):
                pass

            rewards.append(reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path.replace(".txt", "_reasoning_vqa.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Reasoning VQA reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Reference steps: {ref_steps}\n")

        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "vqa":
                    return Qwen2VLModule.vqa_accuracy_reward
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        elif func == "format":
            match task_type:
                case "vqa":
                    return Qwen2VLModule.format_reward_vqa
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        elif func == "reasoning":
            match task_type:
                case "vqa":
                    return Qwen2VLModule.vqa_reasoning_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func} for task type: {task_type}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
