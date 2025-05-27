import torch
import torchvision.transforms as T
from PIL import Image
import time
import json
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_NAME = "5CD-AI/Vintern-1B-v3_5"
DEFAULT_IMAGE_SIZE = 448
DEFAULT_MAX_NUM = 4
DEFAULT_QUESTION = "Trích xuất số, tên, ngày sinh, nguyên quán, ĐKHK trong ảnh. Trả về JSON."


# Image preprocessing functions
def build_transform(input_size=DEFAULT_IMAGE_SIZE):
    """Build image transformation pipeline"""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest aspect ratio from target ratios"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=DEFAULT_MAX_NUM, image_size=DEFAULT_IMAGE_SIZE, use_thumbnail=True):
    """Dynamically preprocess an image based on its aspect ratio"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=DEFAULT_IMAGE_SIZE, max_num=DEFAULT_MAX_NUM):
    """Load and preprocess image from file path"""
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        # Assume image_file is already a PIL Image
        image = image_file

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)


# Device detection and selection
def get_device():
    """Detect and return the appropriate device for computation"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# Model loading functions
def load_model(model_name=MODEL_NAME, device=None):
    """Load model and move to appropriate device"""
    if device is None:
        device = get_device()

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval().to(device)

    return model


def load_tokenizer(model_name=MODEL_NAME):
    """Load tokenizer for the model"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False
    )
    return tokenizer


# Text extraction function
def extract_text_from_image(
    image,
    model,
    tokenizer,
    device=None,
    question=DEFAULT_QUESTION,
    max_num=DEFAULT_MAX_NUM,
    input_size=DEFAULT_IMAGE_SIZE
):
    """
    Extract text from image using the loaded model

    Args:
        image: Path to image file or PIL Image object
        model: Loaded model
        tokenizer: Loaded tokenizer
        device: Computation device
        question: Question to guide the extraction
        max_num: Maximum number of image blocks
        input_size: Input image size

    Returns:
        response: Text response from model
        processing_time: Time taken to process the image
    """
    if device is None:
        device = get_device()

    start_time = time.time()

    # Load and process the image
    pixel_values = load_image(image, input_size=input_size, max_num=max_num).to(
        torch.bfloat16).to(device)

    # Estimate token length
    text_token_ids = tokenizer.encode(question, return_tensors='pt')
    estimated_input_tokens = text_token_ids.shape[-1] + pixel_values.shape[0]
    max_total_tokens = 1700
    max_new_tokens = min(512, max_total_tokens - estimated_input_tokens)

    # Configure generation parameters
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=3,
        repetition_penalty=2.5
    )

    # Generate response
    response, _ = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=True
    )

    end_time = time.time()
    processing_time = end_time - start_time

    return response, processing_time


def parse_json_response(response):
    """Try to parse response as JSON, return as dict if successful"""
    try:
        print(response)
        # Kiểm tra và loại bỏ ký tự "_" ở đầu nếu có
        if response.startswith('_'):
            response = response[1:]
        return json.loads(response)
    except json.JSONDecodeError:
        return {"result": response}


# Example usage (only executed when script is run directly)
if __name__ == "__main__":
    # Set up the model and tokenizer
    device = get_device()
    model = load_model(device=device)
    tokenizer = load_tokenizer()

    # Example image and question
    test_image = 'cccd.jpg'
    question = DEFAULT_QUESTION

    # Process the image
    response, processing_time = extract_text_from_image(
        test_image, model, tokenizer, device, question
    )

    # Parse response as JSON if possible
    json_response = parse_json_response(response)

    # Print results
    print(f'User: {question}')
    print(f'Assistant: {response}')
    print(f'Thời gian xử lý: {processing_time:.2f} giây')

    # Print as formatted JSON if applicable
    if "result" not in json_response:
        print("JSON Response:")
        print(json.dumps(json_response, indent=2, ensure_ascii=False))
