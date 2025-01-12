import os
from dotenv import load_dotenv
from PIL import Image
from src.utils import (
    get_image_embedding,
    get_text_embedding
)
from src.search_embeddings import search_similar_images
from src.gpt_integration import prepare_multimodal_messages
from openai import OpenAI

load_dotenv(dotenv_path=".env")
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)  

def main_workflow(user_image=None, user_text_query=None, top_k=5):
    results = []

    """
    User can input an image or text query, or both, to get fashion recommendations.
    If both text & image are provided, separate searches will be performed and top results merged. 
    """
    
    # Generate image embedding and search, if provided
    if user_image:
        image_embedding = get_image_embedding(user_image)
        image_results = search_similar_images(image_embedding, top_k=top_k)
        results.extend(image_results)

    # Generate text embedding and search, if provided
    if user_text_query:
        text_embedding = get_text_embedding(user_text_query)
        text_results = search_similar_images(text_embedding, top_k=top_k)
        results.extend(text_results)

    # Remove duplicates and retain the top-k unique results
    unique_results = list({(name, image_url) for name, image_url in results})[:top_k]

    query = user_text_query if user_text_query else "I uploaded an image."
    messages = prepare_multimodal_messages(query, unique_results)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )

    gpt_content = response.choices[0].message.content

    image_urls = []
    for name, image_url in unique_results:
        image_urls.append(f"{name} - {image_url}")

    return {
        "gpt_content": gpt_content,
        "image_urls": image_urls
    }

user_image = Image.open("data/sample-dress.jpg")
output = main_workflow(user_image=user_image)