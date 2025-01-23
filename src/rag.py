import os
import io
import hashlib
from typing import List, Optional, Union

import chromadb
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

class PDFVectorStore:
    def __init__(
        self, 
        collection_name: str = "pdf_collection", 
        persist_directory: str = ".vectorstore"
    ):
        """
        Initialize a vector store for PDF documents
        
        :param collection_name: Name of the collection in the vector database
        :param persist_directory: Directory to persist vector database
        """
        # Ensure persistence directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _generate_document_id(self, pdf_path: str, page_number: int) -> str:
        """
        Generate a unique ID for each document page
        
        :param pdf_path: Path to the PDF
        :param page_number: Page number
        :return: Unique document ID
        """
        return hashlib.md5(f"{pdf_path}_page_{page_number}".encode()).hexdigest()

    def add_pdf(
        self, 
        pdf_path: str, 
        chunk_size: int = 300, 
        overlap: int = 50
    ):
        """
        Add a PDF to the vector store
        
        :param pdf_path: Path to the PDF file
        :param chunk_size: Number of tokens per chunk
        :param overlap: Number of tokens to overlap between chunks
        """
        # Read PDF
        reader = PdfReader(pdf_path)
        
        # Process each page
        for page_num, page in enumerate(reader.pages):
            # Extract text
            text = page.extract_text()
            
            # Chunk the text
            chunks = self._chunk_text(text, chunk_size, overlap)
            
            # Add chunks to vector store
            embeddings = self.embedding_model.encode(chunks)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc_id = self._generate_document_id(pdf_path, page_num)
                
                self.collection.add(
                    ids=[f"{doc_id}_chunk_{i}"],
                    documents=[chunk],
                    embeddings=[embedding.tolist()],
                    metadatas=[{
                        "source": pdf_path,
                        "page": page_num,
                        "chunk": i
                    }]
                )

    def _chunk_text(
        self, 
        text: str, 
        chunk_size: int = 300, 
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks
        
        :param text: Input text
        :param chunk_size: Number of tokens per chunk
        :param overlap: Number of tokens to overlap
        :return: List of text chunks
        """
        # Simple tokenization (can be replaced with more sophisticated method)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks

    def search(
        self, 
        query: str, 
        n_results: int = 3
    ) -> List[dict]:
        """
        Search the vector store for most relevant chunks
        
        :param query: Search query
        :param n_results: Number of results to return
        :return: List of most relevant document chunks
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search in vector store
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(n_results):
            formatted_results.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'page': results['metadatas'][0][i]['page']
            })
        
        return formatted_results