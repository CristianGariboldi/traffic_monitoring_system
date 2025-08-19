from transformers import AutoConfig, AutoProcessor
from transformers.image_utils import load_image
import onnxruntime
import numpy as np

# --- 1. Load Models and Configuration ---

print("Loading models and processor...")
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

# Load the model's configuration and processor from Hugging Face
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Load the local ONNX files into ONNX Runtime sessions for GPU execution
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
vision_session = onnxruntime.InferenceSession("onnx/vision_encoder.onnx", providers=providers)
embed_session = onnxruntime.InferenceSession("onnx/embed_tokens.onnx", providers=providers)
decoder_session = onnxruntime.InferenceSession("onnx/decoder_model_merged.onnx", providers=providers)

# Get necessary config values for the generation loop
num_key_value_heads = config.text_config.num_key_value_heads
head_dim = config.text_config.head_dim
num_hidden_layers = config.text_config.num_hidden_layers
eos_token_id = config.text_config.eos_token_id
image_token_id = config.image_token_id
print("Models loaded successfully.")


# --- 2. Prepare Inputs ---

# Create the input messages. This format allows for both image and text.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image in detail?"}
        ]
    },
]

# Load an image from a URL or a local file path
image_url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image = load_image(image_url)

# Apply the processor to format the inputs correctly
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="np")


# --- 3. Run the Generation Loop ---

print("\nGenerating response...")
# Initialize the KV cache (past_key_values) for the decoder
batch_size = inputs['input_ids'].shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}
image_features = None
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# The model expects position_ids, which we can create from the attention_mask
position_ids = np.cumsum(attention_mask, axis=-1) - 1

max_new_tokens = 128
generated_tokens = np.array([[]], dtype=np.int64)

for i in range(max_new_tokens):
    # Get text embeddings for the current input_ids
    inputs_embeds = embed_session.run(None, {'input_ids': input_ids})[0]

    # On the first loop, process the image and merge its features with text embeddings
    if image_features is None:
        # --- FIX: Convert the pixel_attention_mask to boolean type ---
        image_features = vision_session.run(
            None,
            {
                'pixel_values': inputs['pixel_values'],
                'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
            }
        )[0]
        inputs_embeds[inputs['input_ids'] == image_token_id] = image_features.reshape(-1, image_features.shape[-1])

    # Run the decoder model
    logits, *present_key_values = decoder_session.run(None, dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        **past_key_values,
    ))

    # Get the next token by finding the most likely one (argmax)
    next_token_id = logits[:, -1:].argmax(-1, keepdims=False)
    generated_tokens = np.concatenate([generated_tokens, next_token_id], axis=-1)

    # Check if the end-of-sentence token was generated
    if (next_token_id == eos_token_id).all():
        break

    # Update inputs for the next loop iteration
    input_ids = next_token_id
    attention_mask = np.concatenate([attention_mask, np.ones_like(next_token_id)], axis=-1)
    position_ids = np.array([[position_ids[:, -1][0] + 1]])
    
    # Update the KV cache with the new values
    for j, key in enumerate(past_key_values):
        past_key_values[key] = present_key_values[j]

    # (Optional) Print the generated token in real-time (streaming)
    print(processor.decode(next_token_id[0]), end='', flush=True)

print("\n---")


# --- 4. Decode and Print the Final Result ---
final_answer = processor.batch_decode(generated_tokens)[0]
print("\nFinal Answer:", final_answer)