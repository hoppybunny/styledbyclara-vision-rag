import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(dotenv_path=".env")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=pinecone_key)

def search_similar_images(query_embedding, top_k=5):
    """
    Perform similarity search in Pinecone:
    - Query Pinecone index for the top_k results.
    - Retrieve metadata (name, image_url) for matches.
    """
    index = pinecone_client.Index("styledbyclara-index")

    query_results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    results = []
    for match in query_results["matches"]:
        # Ensure metadata exists before accessing it
        metadata = match.get("metadata", {})
        results.append((metadata.get("name"), metadata.get("image_url")))

    return results
