import PyPDF2
import docx
from typing import List, Dict, Any
import io
import re
import os
from logger import logger
from config import Config

class DocumentProcessor:
    """Process various document types for RAG system."""
    
    def __init__(self):
        self.supported_types = Config.ALLOWED_EXTENSIONS
        logger.info("DocumentProcessor initialized.")
    
    def process_uploaded_files(self, uploaded_files_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple uploaded files from their in-memory data."""
        processed_docs = []
        
        for file_data in uploaded_files_data:
            file_name = file_data['name']
            file_type = file_data['type']
            file_bytes = file_data['data']
            
            try:
                logger.info(f"Processing file: {file_name}")
                content = self.extract_text_from_file(file_name, file_bytes)
                if content:
                    chunks = self.chunk_text(content, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
                    for i, chunk in enumerate(chunks):
                        processed_docs.append({
                            'content': chunk,
                            'metadata': {
                                'filename': file_name,
                                'file_type': file_type,
                                'chunk_id': i,
                                'source': 'upload'
                            }
                        })
                logger.info(f"Successfully processed {file_name}, created {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
        
        return processed_docs
    
    def extract_text_from_file(self, file_name: str, file_bytes: bytes) -> str:
        """Extract text from file bytes."""
        file_extension = os.path.splitext(file_name.lower())[1]
        
        if file_extension == '.pdf':
            return self.extract_from_pdf(file_name, file_bytes)
        elif file_extension in ['.txt', '.md']:
            return self.extract_from_text(file_name, file_bytes)
        elif file_extension == '.docx':
            return self.extract_from_docx(file_name, file_bytes)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return ""
    
    def extract_from_pdf(self, file_name: str, file_bytes: bytes) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_name}: {e}")
            return ""

    def extract_from_text(self, file_name: str, file_bytes: bytes) -> str:
        try:
            return file_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading text file {file_name}: {e}")
            return ""

    def extract_from_docx(self, file_name: str, file_bytes: bytes) -> str:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX {file_name}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
            
        return chunks