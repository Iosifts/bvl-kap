import json
import os
import subprocess
import sys
import tempfile
import time

from termcolor import colored

from util import (
    _debug_print_value,
    gather_images,
    get_conf,
    parse_image_dataset,
    parse_grounding_data,
    read_grounding_data,
    unify_paths,
)


def get_prompt(prompt_file):
    prompt_file = prompt_file or get_conf("DEFAULT_PROMPT_FILE")
    with open(prompt_file, "r") as r:
        prompt = r.read()
    return prompt


def _build_fewshot_prefix(fewshot_path):
    """Load BVL few-shot exemplars and format them as a prompt prefix."""
    with open(fewshot_path) as f:
        exemplars = json.load(f)

    parts = ["Here are examples of good BVL descriptions:\n"]
    for ex in exemplars[:2]:  # use first 2 exemplars to keep prompt reasonable
        title = ex.get("title", "Untitled")
        artist = ex.get("artist", "Unknown")
        sentences = [clause["text"] for clause in ex["output"]]
        text = " ".join(sentences)
        parts.append(f"{title} by {artist}:\n{text}\n")
    parts.append("Now describe this artwork:\n")
    return "\n".join(parts)


def interactive_generate_captions(args, debug=True):
    image_paths = gather_images(args.images, max_depth=None)
    base_path, image_names = unify_paths(args.base_path, image_paths)
    prompt = get_prompt(getattr(args, "prompt_file", None))

    # grounding data
    grounding_data = None
    if getattr(args, "grounding_data", None) is not None:
        grounding_data = parse_grounding_data(
            read_grounding_data(args.grounding_data))

    # few-shot prefix
    fewshot_prefix = ""
    fewshot_path = getattr(args, "fewshot", None)
    if fewshot_path is not None:
        fewshot_prefix = _build_fewshot_prefix(fewshot_path)
        print(colored("fewshot: enabled", "cyan")
              + f" ({fewshot_path})")

    timeout = getattr(args, "cache_timeout", None)

    # build reverse mapping: absolute path -> relative name (as in grounding CSV)
    path_to_name = dict(zip(image_paths, image_names))

    if debug:
        arguments = {
            "args": args,
            "unify paths": "\n- ".join(image_paths),
            "base path": base_path,
            "images": ", ".join(image_names),
            "prompt": prompt,
        }
        if fewshot_prefix:
            arguments["fewshot"] = "enabled"
        if timeout is not None:
            arguments["cache timeout"] = timeout
        for name, value in arguments.items():
            _debug_print_value(name, value)
        print("generating captions for "
              + colored(f"{len(image_paths)} image(s)", "green") + "...")

    def generate_caption(_image, path):
        # resolve the relative name used as key in grounding data
        image_name = path_to_name.get(path, os.path.basename(path))

        prompt_tail = ""
        if grounding_data is not None:
            if image_name not in grounding_data:
                raise KeyError(
                    f'grounding data for "{image_name}" was not found in '
                    f'"{args.grounding_data}"'
                )
            grounding_values = grounding_data[image_name]
            labels = [
                "Emotional cues",
                "Detected objects and locations",
                "Main colors",
                "Depth / spatial cues",
            ]
            prompt_tail = (
                "\n\nUse the following supporting evidence to improve accuracy. "
                "Treat it as helpful but not infallible. Prefer details consistent "
                "with the image. Do not list metadata directly — integrate only "
                "useful parts naturally.\n\n"
                "Supporting evidence:\n"
            )
            for label, value in zip(labels, grounding_values):
                prompt_tail += f"- {label}: {value}\n"

        full_prompt = fewshot_prefix + prompt + prompt_tail
        cap = subprocess_caption(path, full_prompt)

        if timeout is not None:
            import torch
            print("clearing cache")
            time.sleep(timeout // 2)
            torch.cuda.empty_cache()
            time.sleep(timeout // 2)

        return {"caption": cap}

    parse_image_dataset(
        args.output, list(zip(image_names, image_paths)), args.output_format,
        generate_caption, overwrite="ask",
    )


def subprocess_caption(image_file, prompt):
    qwen_file = os.path.dirname(__file__) + "/qwen.py"
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "caption.txt")
        command = [sys.executable, qwen_file, image_file, prompt, output_file]
        command = [str(arg) for arg in command]
        print(f"running: {' '.join(command)}")
        subprocess.run(command, text=True, check=True)
        with open(output_file, "r") as fp:
            return fp.read().strip()
