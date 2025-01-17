# Visual RAG for Fashion AI Assistant

## Purpose
The goal is to enable users to find personalized fashion recommendations by uploading an image or entering a text query. The system searches through a Pinecone database of 400 real-world fashion images with associated metadata and generates relevant suggestions enhanced with OpenAI's language capabilities.

### Key features
1. **Visual search:** Users can upload an image to find similar fashion items.
2. **Text based recommendations:** Users can input textual descriptions to get relevant results.
3. **Multimodal:** The system supports combining image and text inputs for richer results.

---

## Technical details

### 1. **Database**
- Pinecone vector database.
- 400 fashion images with metadata (name, image url)
- Embeddings: Images are embedded into a 512-dimensional vector space using OpenAI's CLIP model (`openai/clip-vit-base-patch32`).

### 2. **Data preparation**
- Image scraping
- Embedding generation:
  - Each image is processed using the CLIP model to generate embeddings.
  - Metadata is added directly to Pinecone for retrieval.

### 3. **System architecture**
- Backend: Python-based implementation.
- Search: Pinecone for similarity search.
- OpenAI's GPT-4 for contextual augmentation.

### 4. **Workflow**
1. Images are downloaded, processed into embeddings, and upserted into Pinecone with metadata.
2. Users can input an image, text, or both.
3. Relevant embeddings are generated and used to query the Pinecone index.
4. GPT-4 provides stylistic and contextual recommendations based on search results. Results include product names, links, and image previews.

---

### Example

#### Input
- Image: Upload an image of a dress. (sample image is stored in `data/clothes_data.json`)

#### GPT output
```
Here are some stylish outfit options from StyleNow that you can consider: 

Dance At Dusk Wide Leg Pant: Pair these with a fitted top or blouse to balance the flowy silhouette.

Sunni Midi Dress**: Elegant for both day and evening events, this dress can be accessorized with minimal jewelry and strappy sandals.

Divya Pintuck Tux Shirt: For a bold look, wear this sheer shirt with high-waisted trousers or a pencil skirt. 

Wide Leg Pleated Pant: These pants are perfect for a chic and relaxed look. Pair with a tucked-in blouse or crop top. 

Alberta Dress**: Ideal for formal occasions, this gown would look stunning with statement earrings and sleek heels.

Feel free to mix and match according to your style preferences!', 

'image_urls': ['Dance At Dusk Wide Leg Pant - https://is4.revolveassets.com/images/p4/n/tv/FREE-WP485_V1.jpg', 'Sunni Midi Dress - https://is4.revolveassets.com/images/p4/n/tv/FREE-WD2815_V1.jpg', 'Divya Pintuck Tux Shirt - https://is4.revolveassets.com/images/p4/n/tv/LAGR-WS512_V1.jpg', 'Wide Leg Pleated Pant - https://is4.revolveassets.com/images/p4/n/tv/MBRU-WP17_V1.jpg', 'Alberta Dress - https://is4.revolveassets.com/images/p4/n/tv/VBRD-WD175_V1.jpg'

```

## Extension - using ColPali
ColPali is a document retrieval model that uses the power of Vision Language Models (VLMs) to efficiently index and retrieve information from documents based solely on their visual features. This removes the need for Optical Character Recognition (OCR) and layout detection as well as complex data ingestion pipelines for PDFs. Colpali operates directly on document images. 

In our use case, Colpali could be applied to fashion catalogs. It could index entire fashion lookbooks or catalogs as-is, preserving the visual richness and contextual relationships between text and images.

However, there were several challenges:
1. Embeddings generated by ColPali were extremely high-dimensional (~99,640 dimensions).
2. Large model requiring GPU to run. 