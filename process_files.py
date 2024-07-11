import asyncio
import argparse
import weaviate
import os
from dotenv import load_dotenv
from txt_semantic_processor import TxtSemanticProcessor
from pdf_semantic_processor import PDFSemanticProcessor

from weaviate.classes.config import Configure, Property, DataType

load_dotenv()
WEAVIATE_INSTANCE_URL = os.getenv("WEAVIATE_INSTANCE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Connect to Weaviate cloud instance
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_INSTANCE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY")
    }
)

# Define schema for the chunks
collection = client.collections.create(
    "kb_001",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),
    properties=[
        Property(name="source", data_type=DataType.TEXT, skip_vectorization=True),
        Property(name="content", data_type=DataType.TEXT),
        Property(name="metadata", data_type=DataType.TEXT),
        Property(name="knowledge_base_id", data_type=DataType.TEXT, skip_vectorization=True),
        Property(name="chunk_index", data_type=DataType.INT, skip_vectorization=True),
    ]
)

print("kb collection created")

kb = client.collections.get("kb_001")

for item in kb.iterator():
    print(item.uuid, item.properties)

# Function to upload chunks to Weaviate
def upload_chunks(chunks, source):
    with kb.batch.dynamic() as batch:
        for index, chunk in enumerate(chunks):
            properties = {
                "source": source,
                "content": chunk,
                "metadata": "{}",
                "knowledge_base_id": "kb_001",
                "chunk_index": index
            }
            batch.add_object(properties=properties)

async def process_and_upload_text(file_path):
    # Initialize the text processor
    processor = TxtSemanticProcessor(buffer_size=1)
    result = await processor.process(file_path)
    chunks = result['content']
    upload_chunks(chunks, file_path)
    # Write chunks to a text file
    with open('text_chunks.txt', 'w', encoding='utf-8') as file:
        for chunk in chunks:
            file.write(chunk + "\n\n---\n\n")

async def process_and_upload_pdf(file_path):
    # Initialize the PDF processor
    processor = PDFSemanticProcessor(buffer_size=1)
    result = await processor.process(file_path)
    chunks = result['content']
    upload_chunks(chunks, file_path)
    # Write chunks to a text file
    with open('pdf_chunks.txt', 'w', encoding='utf-8') as file:
        for chunk in chunks:
            file.write(chunk + "\n\n---\n\n")

# Main function
async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process and upload PDF file to Weaviate.')
    parser.add_argument('pdf_file', type=str, help='The path to the PDF file to process')
    args = parser.parse_args()

    # Process and upload the specified PDF file
    await process_and_upload_pdf(args.pdf_file)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

# Close Weaviate client
print("Close Client")
client.close()