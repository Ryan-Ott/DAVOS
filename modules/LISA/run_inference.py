import os
import sys
from argparse import ArgumentParser

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def load_model_and_processor(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name)
    return processor, model

def run_inference(processor, model, image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    segmentation_mask = processor.decode(outputs[0], skip_special_tokens=True)
    return segmentation_mask

def main(args):
    model_name = args.model_name
    image_dir = args.image_dir
    output_dir = args.output_dir
    prompts = args.prompts.split(',')

    os.makedirs(output_dir, exist_ok=True)
    processor, model = load_model_and_processor(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model.to(device)

    for image_file in os.listdir(image_dir):
        if image_file.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(image_dir, image_file)
            for prompt in prompts:
                mask = run_inference(processor, model, image_path, prompt)
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_{prompt.replace(' ', '_')}.png")
                mask.save(output_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output masks")
    parser.add_argument("--prompts", type=str, required=True, help="Comma-separated list of prompts for segmentation")

    args = parser.parse_args()
    main(args)
