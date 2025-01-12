import os
from dotenv import load_dotenv
import numpy as np
import json
from pinecone import Pinecone

from src.utils import download_image, get_clip_model_and_processor, get_image_embedding, get_text_embedding

# Load data
with open("data/clothes_data.json", "r") as f:
    data = json.load(f)

# Init CLIP model and Pinecone
load_dotenv(dotenv_path=".env")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=pinecone_key)
clip_model, clip_processor = get_clip_model_and_processor("openai/clip-vit-base-patch32")

index_name_combined = "styledbyclara-index"
index = pinecone_client.Index(index_name_combined)

failed_links = []

for idx, item in enumerate(data):
    try:
        image_url = item["image_url"]
        name = item["name"]

        image = download_image(image_url)
        if image is None:
            failed_links.append(image_url)
            continue

        image_embedding = get_image_embedding(image).flatten().tolist() 
        index.upsert([
            {
                "id": str(idx),
                "values": image_embedding,
                "metadata": {
                    "name": name,
                    "image_url": image_url
                }
            }
        ])
        print(f"Successfully added embedding for {name}.")

    except Exception as e:
        print(f"Failed to process image {idx}: {str(e)}")
        failed_links.append(image_url)

print(f"Failed links: {len(failed_links)}")
print("Embeddings stored in Pinecone.")


