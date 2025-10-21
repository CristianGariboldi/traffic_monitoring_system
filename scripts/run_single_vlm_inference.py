import argparse
import torch
from PIL import Image
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from llava_next.llava.model.builder import load_pretrained_model
from llava_next.llava.mm_utils import process_images, tokenizer_image_token
from llava_next.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava_next.llava.conversation import conv_templates
from llava_next.llava.utils import disable_torch_init

def query_llava(image_obj, question, tokenizer, model, image_processor):
    """ The same query function we used before. """
    disable_torch_init()
    image_tensor = process_images([image_obj], image_processor, model.config)
    image_tensor = [t.to(dtype=torch.float16, device=model.device) for t in image_tensor]
    image_sizes = [image_obj.size]
    conv = conv_templates["chatml_direct"].copy()
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, images=image_tensor, image_sizes=image_sizes,
            do_sample=False, temperature=0, max_new_tokens=512, use_cache=True
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava-v1.6-34b", load_4bit=True, device_map="auto", attn_implementation=None
    )

    image = Image.open(args.image_file).convert('RGB')
    response = query_llava(image, args.question, tokenizer, model, image_processor)

    print(response)

if __name__ == "__main__":
    main()