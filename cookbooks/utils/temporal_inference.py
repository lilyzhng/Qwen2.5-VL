import os
from openai import OpenAI

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def temporal_inference(model, processor, video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, 
                "total_pixels": total_pixels, 
                "min_pixels": min_pixels},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


def temporal_inference_with_api(
    video_path,
    prompt,
    sys_prompt = "You are a helpful assistant.",
    model_id = "qwen-vl-max-latest",
):
    client = OpenAI(
        api_key = os.getenv('DASHSCOPE_API_KEY'),
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )    
    messages = [
        {
            "role": "system",
            "content": [{"type":"text","text": sys_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_path}},
                {"type": "text", "text": prompt},
            ]
        }
    ]
    completion = client.chat.completions.create(
        model = model_id,
        messages = messages,
    )
    print(completion)
    return completion.choices[0].message.content