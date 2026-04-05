import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_NAME = "prithivMLmods/Qwen3-VL-4B-Instruct-Unredacted-MAX"
# MODEL_NAME = "prithivMLmods/Qwen3-VL-2B-Instruct-Unredacted-MAX-FP8"


def caption(image, prompt):
    torch.cuda.empty_cache()

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()  # TODO added to original
        else torch.float32  # TODO added to original
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",  # TODO original: "cuda" if supported else "cpu"
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    messages = [
        {
            "role": "user",
            "content": [
                # {"type": "image", "image": f"file://{image_path}"},  # TODO original loads Image
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    #images, videos = process_vision_info(messages, image_patch_size=16)

    inputs = processor(
        text=[text],
        images=[image],
        # do_resize=False,  # TODO not in original
        return_tensors="pt",
        padding=True,
    )

    # Move batch to model device
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # TODO not in original
            # TODO original: "use_cache=True,"
            # TODO original: "temperature=1.5,"
            # TODO original: "min_p=0.1,"
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) < 4:
        print("Usage: python run_qwen3vl_min.py IMAGE PROMPT OUTPUT")
        sys.exit(1)

    image_path = os.path.abspath(sys.argv[1])
    prompt = sys.argv[2]

    with open(sys.argv[3], "w") as fp:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        else:
            image = Image.open(image_path).convert("RGB")

        fp.write(caption(image, prompt))
