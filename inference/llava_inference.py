from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests


def setup_llava_model_and_processor(model_path="liuhaotian/llava-v1.6-vicuna-13b", device="cuda"):
    """
    Load the Llava model and processor from the specified path.
    """
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    return model.to(device), processor


def inference_llava_one_step(
    model,
    processor,
    image_path=None,
    user_prompt=None,
    system_prompt=None,
    device="cuda"
):
    if device is not None:
        model.to(device)
    
    if image_path is not None:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "image", "url": image_path}, {"type": "text", "text": user_prompt}]
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ]
    # print("Messages for Llava model:", messages)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
    
    # Generate the output
    output = model.generate(**inputs, max_new_tokens=1024)
    return_str = processor.decode(output[0], skip_special_tokens=True)
    answer_by_model = return_str.split("ASSISTANT: ")[-1].strip()
    
    return answer_by_model


if __name__ == "__main__":
    # Example usage
    model_path = "liuhaotian/llava-v1.6-vicuna-13b"
    model, processor = setup_llava_model_and_processor(model_path)
    
    image_url = "yollava-data/test/bo/0.png"  
    user_prompt = "What is in this image?"
    system_prompt = "You are a helpful assistant."
    
    # Perform inference
    answer = inference_llava_one_step(
        model=model,
        processor=processor,
        image_path=image_url,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        device="cuda"
    )
    
    print("Answer by model:", answer)
