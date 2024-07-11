import re
from semantic_chunker import SemanticChunker
from embeddings import Embeddings

class TxtSemanticProcessor:
    def __init__(self, buffer_size=1):
        self.embeddings = Embeddings()
        self.chunker = SemanticChunker(embeddings=self.embeddings, buffer_size=buffer_size)

    async def process_text(self, text):
        text = self.clean(text)
        chunks = self.chunker.split_text(text)
        return {"status": "success", "type": "txt", "content": chunks}

    async def process(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return await self.process_text(text)

    def clean(self, text: str):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,?!]", "", text)
        text = text.strip()
        return text
