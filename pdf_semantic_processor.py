import io
import fitz  # PyMuPDF
from txt_semantic_processor import TxtSemanticProcessor
from embeddings import Embeddings

class PDFSemanticProcessor:
    def __init__(self, buffer_size=1):
        self.embeddings = Embeddings()
        self.txt_processor = TxtSemanticProcessor(buffer_size=buffer_size)

    async def process(self, file_path):
        with open(file_path, 'rb') as file:
            file_stream = io.BytesIO(file.read())
        text = self.extract_text_from_pdf(file_stream)
        return await self.txt_processor.process_text(text)

    def extract_text_from_pdf(self, file_stream):
        text = ""
        with fitz.open(stream=file_stream, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
