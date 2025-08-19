import os
import json
import torch
from PIL import Image
import re
import sys
from typing import Any, Dict, Union

from s_model_inference import inference_s_model_one_step, load_lora_adapter, setup_model_and_processor, CONCEPT_DESCRIPTIONS
from llava_inference import inference_llava_one_step, setup_llava_model_and_processor

sys.path.append("./eval")
from yollava_eval_s_model import parse_to_json


ORIGINAL_LLAVA_SYSTEM_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."



class SQwenLLLaVARethinkModel:
    def __init__(self,
                concepts,   # "a" or ["a", "b"]
                top_k,
                qwen_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
                lora_base_path="meta-lora-adapters",
                concept_match_file="merged_concept_matches.json",
                rule_for_multi_concept="mean_for_each",
                llava_model_path="liuhaotian/llava-v1.6-vicuna-13b",
                device="cuda",
                ):
        self.concepts = concepts if isinstance(concepts, list) else [concepts]
        self.top_k = top_k
        self.qwen_model_path = qwen_model_path
        self.lora_base_path = lora_base_path
        self.concept_match_file = concept_match_file
        self.rule_for_multi_concept = rule_for_multi_concept
        self.llava_model_path = llava_model_path
        self.device = device

        # print("[INIT] Loading Qwen S-model …")
        self.qwen_model, self.qwen_processor = setup_model_and_processor(
                                                                    model_path=self.qwen_model_path,
                                                                    device=self.device)
        load_result = load_lora_adapter(
            model=self.qwen_model,
            concepts=self.concepts,
            concept_matches_file=self.concept_match_file,
            lora_base_path=self.lora_base_path,
            top_k=self.top_k,
            device=self.device,
            rule_for_multi_concept=self.rule_for_multi_concept,
        )
        if not load_result:
            raise ValueError(f"Failed to load LoRA adapter for concepts: {self.concepts}")
        # print("[INIT] Qwen S-model ready.")

        # print("[INIT] Loading LLaVA L-model …")
        self.llava_model, self.llava_processor = setup_llava_model_and_processor(
            model_path=self.llava_model_path,
            device=self.device,
        )
        # print("[INIT] LLaVA L-model ready.\n")


    def _build_user_prompt(self, concept_dict: dict[str, str]) -> str:
        lines = ["[LIST]"]
        for cid, txt in concept_dict.items():
            lines.append(f"{cid}: {txt}")
        # lines.append("Extract now.")
        return "\n".join(lines)
    

    def parse_concept_info_to_indentity_attributes(self, concept_situation_json):
        """
        concept_situation_json:
        {
            "<bo>": "<bo> is a cute golden retriever puppy. He aways has a playful expression on his face.",
            "<shiba-sleep>": "<shiba-sleep> is a shiba inu sleeping peacefully. He lives in a cozy home with his owner.",
        }

        returns:
        {
            "<bo>": {"category": "a cute golden retriever puppy", "attributes": "always playful expression"},
            "<shiba-sleep>": {"category": "a shiba inu", "attributes": "can sleep peacefully; lives in cozy home with owner"},
        }
        """

        system_prompt = """
You are an information extractor.

Task: Inspect the textual descriptions below and return ONE JSON object that covers EVERY concept in the Concept List while conforming to the schema:
  • "category" - permanent class (should be string, like "a golden retriever puppy", "a blue cartoon character")
  • "attributes" - mutable traits (should be string, like "always playful expression; dresses in trendy clothes")

Return ONE JSON object covering ALL concept IDs, no markdown fences, each Concept ID must stay wrapped in angle brackets (< >).

Example:
user prompt:
[LIST]
<bo>: <bo> is a cute golden retriever puppy with a playful expression.
<shiba-sleep>: <shiba-sleep> is a shiba inu sleeping peacefully in a cozy home.

output:
{
    "<bo>": {
        "category": "a golden retriever puppy",
        "attributes": "always playful expression"
    },
    "<shiba-sleep>": {
        "category": "a shiba inu", 
        "attributes": "can sleep peacefully; lives in cozy home"
    }
}

Rules:
- Output plain English text only, no Markdown, no code fences
- Keep every concept ID enclosed in angle brackets (< >)
- Provide exactly the two keys for each concept, no extras
- Do not add comments or explanatory text outside the JSON object
""".strip()

        user_prompt = self._build_user_prompt(concept_situation_json)

        raw = inference_llava_one_step(
            self.llava_model, self.llava_processor,
            image_path=None,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            device=self.device,
        )

        # print("[PARSE] Raw extractor output:\n", raw)
        big_json = parse_to_json(raw)
        # print("[PARSE] Parsed JSON:\n", big_json)

        result = {}
        for cid in concept_situation_json:
            if cid in big_json and \
            all(k in big_json[cid] for k in ("category", "attributes")):
                result[cid] = big_json[cid]
            else:
                result[cid] = {"category": "A character or object.", "attributes": CONCEPT_DESCRIPTIONS.get(cid[1:-1],"")}
        
        return result


    def inference_text_only(self,
                            system_prompt, 
                            user_prompt):
        
        # step1 
        concept_situation_json = {f"<{concept}>": CONCEPT_DESCRIPTIONS.get(concept, "") for concept in self.concepts}
        identity_attributes_for_concepts = self.parse_concept_info_to_indentity_attributes(concept_situation_json)
        for concept in concept_situation_json.keys():
            concept_situation_json[concept] = identity_attributes_for_concepts.get(concept,
                {"category": "A character or object.", 
                 "attributes": CONCEPT_DESCRIPTIONS.get(concept[1:-1], "")})
        # print("[TEXT] Concept list after parsing:\n", concept_situation_json)

        # step2 
        # 2.1 
        concept_blocks = []
        for cid, info in concept_situation_json.items():
            concept_blocks.append(
                f"{cid}: category=\"{info['category']}\"; "
                f"attributes=\"{info['attributes']}\""
            )
        concept_list_text = "\n".join(concept_blocks)

        # 2.2 
        merged_system_prompt = (
            system_prompt.strip()
            + "\n[CONCEPT LIST]\n"
            + concept_list_text
            + "\n'category' is the permanent essence of the concept, like 'a cute golden retriever puppy'."
            + "\n'attributes' is the mutable outward traits of the concept, like 'always playful expression'."
            + "\nRules: Use [CONCEPT LIST] to answer the question. If a concept is mentioned in the question, you should use the information in [CONCEPT LIST] to answer it.\n"
        )

        # 2.3 
        answer = inference_llava_one_step(
            model=self.llava_model,
            processor=self.llava_processor,
            image_path=None,                 # text-only
            user_prompt=user_prompt,
            system_prompt=merged_system_prompt,
            device=self.device,
        )
        # print("[TEXT] Final answer:\n", answer)

        return {
            "final_answer": answer.strip(),
            "detection_report_final": concept_situation_json  # 包含每个 concept 的检测结果
        }


    def inference_vqa(self,
                    image_path, 
                    system_prompt, 
                    user_prompt):
        
        # step1 
        # print("[VQA] Running S‑model detection …")
        qwen_model_response = inference_s_model_one_step(
                concepts=self.concepts,
                image_path=image_path,
                model=self.qwen_model,
                processor=self.qwen_processor,
                device=self.device,)
        # print("[VQA] Raw S‑model output:\n", qwen_model_response)

        try:
            qwen_model_response_json = parse_to_json(qwen_model_response["response"])
            # anwser_by_model_json_for_this_concept = answer_by_model_json.get(f"<{concept_to_eval}>", "")
            for concept, situation in qwen_model_response_json.items():
                if situation["present"]:
                    situation["location-absolute"] = situation.get("location-absolute", "")
                    situation["location-relative"] = situation.get("location-relative", "")
                    situation["concept-info"] = CONCEPT_DESCRIPTIONS.get(concept[1:-1], situation.get("concept-info", ""))
                else:
                    situation["location-absolute"] = ""
                    situation["location-relative"] = ""
                    situation["concept-info"] = ""
            s_model_json = qwen_model_response_json
        except Exception as e:
            print(f"Error parsing JSON for concepts in image {image_path}: {e}")
            s_model_json = {f"<{concept}>": {"present": False, "location-absolute": "", "location-relative": "", "concept-info": ""} for concept in self.concepts}
        
        # print("[VQA] Parsed S‑model JSON:\n", s_model_json)

        # step2 
        subset_json = {k: v["concept-info"] for k, v in s_model_json.items() if v["present"]}

        llava_reflection_respose = {k: "" for k in subset_json.keys()} 
        if not subset_json:
            det_clean = s_model_json
            print("[VQA] No concepts detected in the image. Skipping self-VQA.")
        else:
            identity_attributes_for_subset = self.parse_concept_info_to_indentity_attributes(subset_json)
            for concept, situation in s_model_json.items():
                if situation["present"]:
                    situation["concept-info"] = identity_attributes_for_subset.get(concept, 
                        {"category": "A character or object.", 
                        "attributes": CONCEPT_DESCRIPTIONS.get(concept[1:-1], "")})
            
            ## rethink with s_model_json
            det_clean = {cid: v.copy() for cid, v in s_model_json.items()}

            for cid, v in list(s_model_json.items()):
                if not v["present"]:
                    continue  # 只有 present==true 的才需要验证

                category = v["concept-info"]["category"]
                loc_abs  = v["location-absolute"]
                loc_rel  = v["location-relative"]

                # 1. 
                block_lines = [
                    # f"{cid}: category: {category}",
                    f"Q1. Do you see {category} at {loc_abs}? (yes/no)",
                    f"Q2. Is {category} {loc_rel}? (yes/no)",
                    "Rules: Answer each question strictly yes or no. If you are not sure, answer no."
                    "If there are N Questions, answer N times yes or no, separated by space."
                ]

                single_rethink_prompt = "[SELF-VQA]\n\n" + "\n".join(block_lines)
                print(f"[RETHINK] {cid} self-VQA prompt:\n", single_rethink_prompt)

                # 2. 
                answer = inference_llava_one_step(
                    model=self.llava_model,
                    processor=self.llava_processor,
                    image_path=image_path,
                    user_prompt=single_rethink_prompt,
                    system_prompt = "You are a visual verifier. Answer each visual question with yes or no only. Provide exactly one 'yes' or 'no' per question. If there are N questions, output N tokens separated by a single space. Do not include any additional words, punctuation, or commentary.",
                    device=self.device,
                ).strip().lower()
                llava_reflection_respose[cid] = answer  

                print(f"[RETHINK] {cid} self-VQA raw answer:\n", answer)

                # 3. 
                yn = re.findall(r"\b(yes|no)\b", answer)[:2] + ["no", "no"]
                ans1, ans2 = yn[0], yn[1]

                # 4. 
                if ans1 == "no" and ans2 == "no":
                    det_clean[cid]["present"] = False
                    det_clean[cid]["location-absolute"]  = ""
                    det_clean[cid]["location-relative"]  = ""
                    det_clean[cid]["concept-info"] = ""
                else:
                    if ans1 == "no":
                        det_clean[cid]["location-absolute"] = ""
                    if ans2 == "no":
                        det_clean[cid]["location-relative"] = ""
            
            # print("[RETHINK] det_clean after self-VQA:\n", det_clean)
            
        det_lines = []
        for cid, info in det_clean.items():
            if info["present"]:
                iden  = info["concept-info"]["category"]
                attr  = info["concept-info"].get("attributes", "")
                loc_a = info["location-absolute"]
                loc_r = info["location-relative"]
                det_lines.append(
                    f"{cid}: present=true; location-absolute=\"{loc_a}\"; location-relative=\"{loc_r}\"; category=\"{iden}\"; attributes=\"{attr}\""
                )
            else:
                det_lines.append(f"{cid}: present=false")

        det_report = "\n".join(det_lines)

        # 4-B 
        final_system_prompt = (
            system_prompt.strip() +
            "\n\n[DET_REPORT]\n" + det_report +
            "\n\nRules:\n"
            "Use DET_REPORT to answer the user's visual question.\n"
            "- 'category' is the immutable essence of the concept, like 'a golden retriever puppy'.\n"
            "- 'attributes' is the mutable traits that may or may not be visible, like 'always playful expression'.\n"
            "- If present = false, it means the concept is not in the image. You should not mention the concept; reply 'no' if asked about its presence.\n"
            "- If present = true, it means the concept is in the image. You should ground your answer strictly on the provided fields; reply 'yes' if asked about its presence.\n"
        )

        # print("[VQA] Final system prompt:\n", final_system_prompt)

        # 4-C 
        final_answer = inference_llava_one_step(
            model=self.llava_model,
            processor=self.llava_processor,
            image_path=image_path,
            user_prompt=user_prompt,
            system_prompt=final_system_prompt,
            device=self.device,
        )

        # print("[VQA] Final answer:\n", final_answer)

        return {
            "final_answer": final_answer.strip(),
            "llava_reflection_response": llava_reflection_respose,  
            "detection_report_final": det_clean,  
            "detection_report_raw": s_model_json,  
        }


    def inference(self, 
                image_path, 
                system_prompt, 
                user_prompt):

        # If it is a text-only issue, skip the small model.
        if image_path is None:
            # Text-only inference
            return self.inference_text_only(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        # If it is a VQA question, enter the pipeline
        else:
            return self.inference_vqa(
                image_path=image_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )


if __name__ == "__main__":
    # Example usage
    model = SQwenLLLaVARethinkModel(
        concepts="mam",
        top_k=1
    )

    system_prompt = ORIGINAL_LLAVA_SYSTEM_PROMPT
    user_prompt = "Who is <mam>? Describe him/her in detail."
    image_path = "yollava-data/test/bo/3.png"

    result = model.inference(
        image_path=image_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    print("\n=== OUTPUT ===\n", result)

