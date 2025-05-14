from typing import Dict, List, Optional
import torch
import faiss
import numpy as np
from transformers import MT5EncoderModel, AutoTokenizer, AutoModel
import wikipedia
import os
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class RAG:
    def __init__(
        self,
        model_name: str = "google/mt5-base",
        device: str = None,
        index_path: Optional[str] = None
    ):
        """
        Initialize RAG module.
        
        Args:
            model_name: Name of the encoder model
            device: Device to run on
            index_path: Path to load existing FAISS index
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = MT5EncoderModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize or load FAISS index
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(index_path + '.json', 'r') as f:
                self.documents = json.load(f)
        else:
            self.index = None
            self.documents = []
            
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embeddings.
        
        Args:
            text: Input text
            
        Returns:
            np.ndarray: Text embeddings
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use [CLS] token embedding as text representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
        return embeddings
        
    def build_index(
        self,
        documents: List[Dict[str, str]],
        batch_size: int = 32
    ):
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of documents with text and metadata
            batch_size: Batch size for encoding
        """
        # Encode all documents
        embeddings = []
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i + batch_size]
            batch_texts = [doc["text"] for doc in batch]
            batch_embeddings = self.encode_text(batch_texts)
            embeddings.append(batch_embeddings)
            
        embeddings = np.vstack(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.documents = documents
        
    def search(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict[str, str]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if not self.index:
            return []
            
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return relevant documents
        return [self.documents[i] for i in indices[0]]
        
    def get_wikipedia_context(
        self,
        query: str,
        max_pages: int = 3
    ) -> List[str]:
        """
        Get relevant context from Wikipedia.
        
        Args:
            query: Search query
            max_pages: Maximum number of pages to retrieve
            
        Returns:
            List[str]: Retrieved content chunks
        """
        search_results = wikipedia.search(query, results=max_pages)
        contents = []
        for page_title in search_results:
            try:
                page = wikipedia.page(page_title)
                contents.append(page.content)
            except wikipedia.exceptions.DisambiguationError as e:
                # If the page is a disambiguation page, skip it
                continue
            except wikipedia.exceptions.PageError:
                # If the page does not exist, skip it
                continue
        return contents
        
    def _get_wikipedia_content(self, query: str, max_pages: int = 3) -> List[str]:
        """
        Retrieve relevant Wikipedia content.
        
        Args:
            query: Search query
            max_pages: Maximum number of pages to retrieve
            
        Returns:
            List[str]: Retrieved content chunks
        """
        # Search Wikipedia
        search_results = self.wiki.search(query, results=max_pages)
        
        contents = []
        for page in search_results.values():
            if page.exists():
                # Get page content and split into chunks
                content = page.text
                chunks = self._split_into_chunks(content)
                contents.extend(chunks)
                
        return contents
        
    def _split_into_chunks(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size
            
        Returns:
            List[str]: Text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 2):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
        
    def _compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            torch.Tensor: Text embeddings
        """
        # Tokenize and encode
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Compute embeddings
        with torch.no_grad():
            outputs = self.encoder(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings
        
    def _find_relevant_chunks(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 3
    ) -> List[str]:
        """
        Find most relevant chunks for a query.
        
        Args:
            query: Search query
            chunks: List of text chunks
            top_k: Number of chunks to retrieve
            
        Returns:
            List[str]: Most relevant chunks
        """
        # Compute embeddings
        query_embedding = self._compute_embeddings([query])
        chunk_embeddings = self._compute_embeddings(chunks)
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.numpy(),
            chunk_embeddings.numpy()
        )[0]
        
        # Get top-k chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [chunks[i] for i in top_indices]
        
    def enrich_generation(self, text: str, style: str) -> str:
        """Enrich generation with relevant knowledge."""
        # Get relevant documents
        docs = self.get_wikipedia_context(text, max_pages=3)
        
        # Add relevant knowledge to the prompt
        if docs:
            knowledge = "\n".join([f"Relevant knowledge: {doc}" for doc in docs])
            return f"{text}\n\n{knowledge}"
        return text
        
    def save_index(self, path: str):
        """Save FAISS index and documents."""
        if self.index:
            faiss.write_index(self.index, path)
            with open(path + '.json', 'w') as f:
                json.dump(self.documents, f)
                
    def load_index(self, path: str):
        """Load FAISS index and documents."""
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            with open(path + '.json', 'r') as f:
                self.documents = json.load(f) 