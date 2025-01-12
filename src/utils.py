import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
import torch
import base64

clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name).eval()
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

def download_image(url):
    try:
        response = requests.get(url, timeout=25)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except requests.RequestException as e:
        print(f"Failed to download image: {url}: {e}")
        return None

def encode_image_to_base64(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_clip_model_and_processor(model_name):
    model = CLIPModel.from_pretrained(model_name).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def get_image_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings.cpu().numpy()

def get_text_embedding(text: str):
    inputs = clip_processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    return embeddings.cpu().numpy()

def get_image_from_base64(base64_str):
    # Remove the data:image/...;base64, part
    base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image