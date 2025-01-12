from src.utils import encode_image_to_base64

def prepare_multimodal_messages(query, top_k_images):
    PROMPT = """
    You are a style assistant from StyleNow helping someone improve their fashion sense.
    Recommendations should match the user's gender. Don't need to tell the user this, just take note yourself. 
    If the gender is unclear, provide generic recommendations.
    Outfits matching the user's style from StyleNow's dataset are shown below.
    If no images are provided, suggest styles and mention no outfits are available in StyleNow's dataset.
    Query: {query}
    """

    messages = [
        {"role": "system", "content": "You are a fashion expert helping a user pick outfits."},
        {"role": "user", "content": [{"type": "text", "text": PROMPT.format(query=query)}]}
    ]

    for name, image_url in top_k_images:
        image_base64 = encode_image_to_base64(image_url)
        messages[-1]["content"].append({"type": "text", "text": name})
        messages[-1]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    return messages