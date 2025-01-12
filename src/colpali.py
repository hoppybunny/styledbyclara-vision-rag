import os
from dotenv import load_dotenv
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pdf2image import convert_from_path
from tqdm import tqdm
from pinecone import Pinecone

# Initialize ColPali model and processor
model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v0.1",
    torch_dtype=torch.float32,
    device_map="mps",  # mps for apple silicon
)
processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

# Initialize Pinecone for storing embeddings
load_dotenv(dotenv_path=".env")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=pinecone_key)
index_name_combined = "colpali-index"
index = pinecone_client.Index(index_name_combined)

def index_document(pdf_path):
    """
    Index a PDF document by converting pages to embeddings and storing them in Pinecone.
    """
    # Convert PDF pages to images
    pages = convert_from_path(pdf_path)
    print(f"PDF contains {len(pages)} pages. Processing...")

    # Iterate through each page and generate embeddings
    for page_number, page_image in enumerate(tqdm(pages, desc="Indexing Pages")):
        try:
            # Generate patch embeddings for the page
            inputs = processor.process_images([page_image]).to(model.device)
            with torch.no_grad():
                page_embeddings = model(**inputs)

            # Reduce dimensionality and prepare metadata
            reduced_embeddings = page_embeddings.to("cpu").numpy().flatten().tolist()

            metadata = {
                "page_number": page_number + 1,
            }

            index.upsert([
                {
                    "id": f"{page_number + 1}",
                    "values": reduced_embeddings,
                    "metadata": metadata,
                }
            ])
            print(f"Indexed page {page_number + 1} of {len(pages)}")
        except Exception as e:
            print(f"Error processing page {page_number + 1}: {e}")


pdf_path = "data/sample-herWorld.pdf" 
index_document(pdf_path)
