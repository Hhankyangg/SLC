import os
import json
import torch
import gc
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from safetensors import safe_open
from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT = """
[SYS]
You are a high-precision concept detector.

[TASK]
You should inspect the image and output **one** JSON object that covers **every** concept in the Concept List provided by the user while conforming to the schema below.

For each concept <concept id> in the list:
- If visible, set:
  "present": true
  "location-absolute": "concise area, e.g. 'top-left quadrant'"  
  "location-relative": "spatial relation, e.g. 'to the left of the person in black suit'"
- If not visible, set:
  "present": false and **omit all other keys**

The final output should be a JSON object like:
{
  "<concept id 1>": {
    "present": <boolean>,
    "location-absolute": <string>,
    "location-relative": <string>
  },
  "<concept id 2>": {
    "present": <boolean>,
    "location-absolute": <string>, 
    "location-relative": <string>
  }
}

Rules:
- Output plain English text only, no Markdown, no code fences
- Keep every concept ID enclosed in angle brackets (< >)
- If present = false, **omit all other keys**
- Boolean literals must be lowercase true/false
- Do **not** add any extra keys, comments, or explanatory text
"""



def setup_model_and_processor(model_path, device):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": device}
    )
    return model, processor


def lora_checksum(m):
    vals = [p.detach().cpu().float().sum() for n, p in m.named_parameters()]
    return torch.stack(vals).sum()


def _load_concept_descriptions(info_file_path="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/Pers/LLaMA-Factory/data/info/info1.json"):
    try:
        with open(info_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {info_file_path}: {e}")
        return {}
CONCEPT_DESCRIPTIONS = _load_concept_descriptions()


def load_lora_adapter(model, 
                    concepts,           # ["a", "b"]
                    concept_matches_file="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/Pers/LLaMA-Factory/data/info/merged_concept_matches.json",    
                    lora_base_path="/ytech_m2v5_hdd/workspace/kling_mm/yangsihan05/Pers/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/lora/new_meta/", 
                    top_k=1, 
                    device="cuda", 
                    rule_for_multi_concept="mean_for_each"):

    def _map_key_to_default(key, adapter_name="default"):
        if ".lora_A." in key and f".lora_A.{adapter_name}." not in key:
            return key.replace(".lora_A.", f".lora_A.{adapter_name}.")
        if ".lora_B." in key and f".lora_B.{adapter_name}." not in key:
            return key.replace(".lora_B.", f".lora_B.{adapter_name}.")
        return key

    def _build_temp_peft_model(base_model, template_dir):
        """
        用 template_dir 下的 adapter_config.json 把 base_model 变成临时 PeftModel
        （仍在 CPU）。兼容 PEFT 各版本：json -> dict -> LoraConfig(**dict)
        """
        cfg_path = os.path.join(template_dir, "adapter_config.json")
        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)
        cfg = LoraConfig(**cfg_dict)
        peft_model = get_peft_model(base_model, cfg) 
        return peft_model
        
    def _average_lora_weights(lora_dirs):
        merged = {}
        for d in lora_dirs:
            wt_path = os.path.join(d, "adapter_model.safetensors")
            if not os.path.exists(wt_path):
                print(f"[WARN] {wt_path} 不存在，跳过")
                raise RuntimeError(f"LoRA 权重文件 {wt_path} 不存在")
            with safe_open(wt_path, framework="pt", device="cpu") as f:
                print(f"[INFO] 正在读取 LoRA 权重：{wt_path}")
                for k in f.keys():
                    merged.setdefault(k, []).append(f.get_tensor(k))
        if not merged:
            raise RuntimeError("所有 LoRA 权重都未成功读取")
        for k, v_list in merged.items():
            merged[k] = torch.stack(v_list, dim=0).mean(dim=0)  # CPU 里做平均

        return merged

    def _copy_into_model(peft_model, merged_dict, tgt_device):
        adapter_name = "default"
        state_dict = peft_model.state_dict()

        miss = []
        with torch.no_grad():
            for src_key, tensor in merged_dict.items():
                tgt_key = src_key if src_key in state_dict else _map_key_to_default(src_key, adapter_name)
                if tgt_key in state_dict:
                    state_dict[tgt_key].copy_(tensor)
                else:
                    miss.append(src_key)
        if miss:
            raise RuntimeError(
                f"未能匹配 {len(miss)} 个 LoRA 张量：{miss[:3]} ...\n"
                "请检查 LoRA 权重文件和模型的兼容性。"
            )
        peft_model.load_state_dict(state_dict, strict=True)

        merged_base = peft_model.merge_and_unload()
        merged_base.to(tgt_device)

        return merged_base

    with open(concept_matches_file, "r") as f:
        concept_matches = json.load(f).get("concept_matches", {})

    # ---------- ① top_k == -1：平均全部 meta LoRA ---------- #
    if top_k == -1:
        fixed_concepts = [
            "zhangwei", "Tongren", "mam", "oong", "Shaoyu",
            "jierui", "Gloria", "HouTeng", "duck-banana", "Hermione"
        ]
        # lora_dirs = [os.path.join(lora_base_path, c) for c in fixed_concepts]
        # load 80 steps checkpoints
        lora_dirs = [os.path.join(lora_base_path, c, "checkpoint-80") for c in fixed_concepts]
        merged = _average_lora_weights(lora_dirs)
        model = _build_temp_peft_model(model, lora_dirs[0])
        model = _copy_into_model(model, merged, device)
        return True

    # ---------- ② 根据输入 concept 选择 top-k ---------- #
    if isinstance(concepts, str):
        concepts = [concepts]

    for c in concepts:
        if c not in concept_matches:
            print(f"[ERR] Concept '{c}' 不在 concept_match 表中")
            return False

    matched_lists = [concept_matches[c] for c in concepts]  # e.g. [["a1","a2"],["b1","b2"]]

    # 这里只实现“单 concept + top-k”
    if len(matched_lists) == 1:
        cand = matched_lists[0]
        if len(cand) < top_k:
            print(f"[ERR] '{concepts[0]}' 仅匹配到 {len(cand)} 个 LoRA (< top_k={top_k})")
            return False
        top_k_concepts = cand[:top_k]
        # print(f"[INFO] 使用 top-{top_k} LoRA：{top_k_concepts}")

        lora_dirs = [os.path.join(lora_base_path, c) for c in top_k_concepts]
        merged = _average_lora_weights(lora_dirs)
        # before = lora_checksum(model)       # copy 前
        model = _build_temp_peft_model(model, lora_dirs[0])
        model = _copy_into_model(model, merged, device)
        # after  = lora_checksum(model)       # copy 后
        # print("checksum before:", before, "after:", after)
        return True

    # ---------- ③ 多 concept 场景 ---------- #
    else:
        if rule_for_multi_concept == "mean_for_each":
            match_concepts =  [sublist[0] if len(sublist) > 0 else None for sublist in matched_lists] # [a1, b1]
            # print(match_concepts)
            lora_dirs = [os.path.join(lora_base_path, c) for c in match_concepts]
            merged = _average_lora_weights(lora_dirs)
            model = _build_temp_peft_model(model, lora_dirs[0])
            model = _copy_into_model(model, merged, device)
            return True
        else:
            multi_concept = "_".join(concepts)
            if multi_concept not in concept_matches:
                print(f"[ERR] '{multi_concept}' 不在 multi_concept 匹配表中")
                return False
            final_concept = concept_matches[multi_concept][0]
            # print(final_concept)
            lora_dirs = [os.path.join(lora_base_path, final_concept)]
            merged = _average_lora_weights(lora_dirs)
            model = _build_temp_peft_model(model, lora_dirs[0])
            model = _copy_into_model(model, merged, device)
            return True
    # raise NotImplementedError("rule_for_multi_concept 尚未实现；请先合并单 concept。")


def inference_s_model_one_step(
    concepts,       # e.g. ["a", "b"]
    image_path,
    model,
    processor,
    device="cuda"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(concepts, str):
        concepts = [concepts]

    system_prompt = SYSTEM_PROMPT + "\n[CONCEPT LIST]\n" + "".join(
        f"- <{cid}>: {CONCEPT_DESCRIPTIONS[cid]}\n" for cid in concepts
    )
    concepts_str = ", ".join(f"<{cid}>" for cid in concepts)

    try:
        image = Image.open(image_path).convert("RGB")
        
        user_prompt = f"""
[TASK]
Inspect the image and produce **one** JSON object that covers **every** concept in:
{concepts_str}

For each concept key (keep it wrapped in < >):

• If the concept **is visible**:
    "present": true,
    "location-absolute": "<concise area, e.g. 'top-left quadrant'>",
    "location-relative": "<its spatial relation to other objects>",
    "concept-info": "<EXACT description copied from [CONCEPT LIST]>"

• If the concept **is NOT visible**:
    "present": false        ← omit the other three keys

Strict rules  
1. Use lowercase true / false.  
2. Do not add extra keys, text, or markdown fences.  
3. Each concept ID must be wrapped in angle brackets (< >).
4. Return the JSON **and nothing else**.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"}, 
                {"type": "text", "text": user_prompt}]}
        ]
        input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoding = processor(text=input_text, images=[image], padding=True, return_tensors="pt")
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model.generate(**encoding, max_new_tokens=1024, do_sample=True, temperature=0.2)
            input_length = encoding["input_ids"].shape[1]
            generated_ids = outputs[0, input_length:]
            response = processor.decode(generated_ids, skip_special_tokens=True)
        
        del encoding, outputs
        
        return {
            "image": image_path,
            "response": response,
        }
        
    except Exception as e:
        print(f"Error during inference for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def inference_s_model_one_step_custom(
    concepts,       # e.g. ["a", "b"]
    image_path,
    system_prompt,
    user_prompt,
    model,
    processor,
    device="cuda"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(concepts, str):
        concepts = [concepts]

    system_prompt = system_prompt

    try:
        if image_path is None:
            #text-only
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoding = processor(text=input_text, padding=True, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = model.generate(**encoding, max_new_tokens=1024, do_sample=True, temperature=0.2)
                input_length = encoding["input_ids"].shape[1]
                generated_ids = outputs[0, input_length:]
                response = processor.decode(generated_ids, skip_special_tokens=True)

            del encoding, outputs

            return {
                "image": None,
                "response": response,
            }
        else:
            image = Image.open(image_path).convert("RGB")
            user_prompt = user_prompt
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"}, 
                    {"type": "text", "text": user_prompt}]}
            ]
            input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoding = processor(text=input_text, images=[image], padding=True, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = model.generate(**encoding, max_new_tokens=1024, do_sample=True, temperature=0.2)
                input_length = encoding["input_ids"].shape[1]
                generated_ids = outputs[0, input_length:]
                response = processor.decode(generated_ids, skip_special_tokens=True)
            
            del encoding, outputs
            
            return {
                "image": image_path,
                "response": response,
            }
        
    except Exception as e:
        print(f"Error during inference for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
if __name__ == "__main__":
    # Example usage
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, processor = setup_model_and_processor(model_path, device)
    success = load_lora_adapter(
        model=model,
        concepts = ["Xiaobai","Xuxian"],
        concept_matches_file="merged_concept_matches.json",
        lora_base_path="meta-lora-adapters",
        top_k=1,
        device="cuda",
        rule_for_multi_concept="avg"
    )
    if not success:
        print("Failed to load LoRA adapter.")
        exit(1)
    print("LoRA adapter loaded successfully.")
