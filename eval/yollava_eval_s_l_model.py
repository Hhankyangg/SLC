import os
import json
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import argparse
import time
from tqdm import tqdm
import sys
from datetime import datetime 

sys.path.append("./inference")
from s_l_model_inference import SQwenLLLaVARethinkModel

import re
from typing import Any, Dict, Union

ORIGINAL_LLAVA_SYSTEM_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."


class YoLLaVAEvalDataset(Dataset):
    def __init__(self, 
                concept_to_eval,
                eval_type: str = "vqa",     # should be "vqa" "qa" "sqa" or "rec"
                json_file="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/Pers/LLaMA-Factory/data/yollava_eval/rec.json"):
        
        assert eval_type in ["vqa", "qa", "sqa", "rec"], f"Unsupported eval_type: {eval_type}"
        self.eval_type = eval_type

        with open(json_file, 'r') as f:
            all_data = json.load(f)
        self.data = all_data.get(concept_to_eval, [])
        self.concept_to_eval = concept_to_eval
        if len(self.data) == 0:
            raise ValueError(f"No data found for concept {concept_to_eval} in {json_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # image_path = item["images"][0]
        image_path = item.get("images", [None])[0]
        system_prompt = item["conversations"][0]["value"]
        user_prompt = item["conversations"][1]["value"].replace("<image>\n", "").strip()

        if self.eval_type == "rec":
            ground_truth = item["is_positive"]
        elif self.eval_type in ["vqa", "qa", "sqa"]:
            ground_truth = item["gt"]
        
        return {
            "image_path": image_path,
            "ground_truth": ground_truth,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        }


def run_s_l_model_evaluation(args, output_dir):

    eval_model = SQwenLLLaVARethinkModel(
        concepts=args.concept_to_eval,
        top_k=args.top_k,
        qwen_model_path=args.qwen_model_path,
        lora_base_path=args.lora_base_path,
        concept_match_file=args.concept_match_file,
        llava_model_path=args.llava_model_path,
        device=args.device,
    )
    print(f"Loaded eval model for concept {args.concept_to_eval} with top_k={args.top_k}.")

    concept_to_eval = args.concept_to_eval
    dataset = YoLLaVAEvalDataset(
        eval_type=args.eval_type,
        concept_to_eval=concept_to_eval,
        json_file=args.test_file)
    print(f"Loaded {args.eval_type} dataset for concept {concept_to_eval} with {len(dataset)} items.")
    
    if args.eval_type == "rec":
        result_sum = {
            "concept": concept_to_eval,
            "yes_recall": 0,
            "no_recall": 0,
            "detail": []}
    else:
        result_sum = {
            "concept": concept_to_eval,
            "accuracy": 0,
            "detail": []}
    
    if args.eval_type == "rec":
        yes_recall_score = 0
        no_recall_score = 0
        yes_total = 0
        no_total = 0
    else:
        correct_count = 0
        total_count = 0


    for item in tqdm(dataset, desc=f"Evaluating concept {concept_to_eval}"):
        image_path = item["image_path"]
        ground_truth = item["ground_truth"]
        user_prompt = item["user_prompt"]

        if args.eval_type == "rec":
            if ground_truth:
                yes_total += 1
            else:
                no_total += 1
        else:
            total_count += 1
        
        response_raw = eval_model.inference(
            image_path=image_path,
            system_prompt=ORIGINAL_LLAVA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        response_str = response_raw["final_answer"].strip().lower()

        if args.eval_type == "rec":
            if ground_truth and "yes" in response_str:
                yes_recall_score += 1
            elif not ground_truth and "no" in response_str:
                no_recall_score += 1
        else:
            if ground_truth.lower() in response_str:
                correct_count += 1
        
        result_sum["detail"].append({
            "image_path": image_path,
            "user_prompt": user_prompt,
            "ground_truth": ground_truth,
            "response": response_raw})
    
    if args.eval_type == "rec":
        result_sum["yes_recall"] = yes_recall_score / yes_total if yes_total > 0 else 0
        result_sum["no_recall"] = no_recall_score / no_total if no_total > 0 else 0
    else:
        result_sum["accuracy"] = correct_count / total_count if total_count > 0 else 0

    with open(f"{output_dir}/{concept_to_eval}.json", "w") as f:
        json.dump(result_sum, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="data/yollava_eval/vqa.json")
    parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--llava_model_path", type=str, default="liuhaotian/llava-v1.6-vicuna-13b")
    parser.add_argument("--lora_base_path", type=str, default="meta-lora-adapters")
    parser.add_argument("--concept_match_file", type=str, default="merged_concept_matches.json")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--concept_to_eval", type=str, default="bo")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_type", type=str, default="rec")   # should be "vqa" "qa" "sqa" or "rec"
    args = parser.parse_args()
    categoriy = os.path.basename(os.path.dirname(args.test_file))
    filename = os.path.splitext(os.path.basename(args.concept_match_file))[0]

    output_dir = f"output/yollava_s_l_model_top-{args.top_k}_eval_type-{args.eval_type}-{filename}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to {output_dir}")
    print(f"Evaluating concept: {args.concept_to_eval}")
    run_s_l_model_evaluation(args, output_dir)
