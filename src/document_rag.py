import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentChunk:
    def __init__(self, content: str, source: str, chunk_id: int, page: Optional[int] = None):
        self.content = content.strip()
        self.source = source
        self.chunk_id = chunk_id
        self.page = page

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "content": self.content
        }


class BusinessDocumentRAG:
    """
    Business Document RAG (Retrieval-Augmented Generation) Engine.
    Ingests PDF, DOCX, TXT, and Markdown documents, performs semantic chunking,
    and retrieves relevant passages with verifiable source citations.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def extract_text(self, file_path_or_bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extract text from file path or file-like buffer based on extension.
        Returns a list of dicts: [{"text": str, "page": Optional[int]}].
        """
        ext = os.path.splitext(filename)[1].lower()
        extracted_pages = []

        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path_or_bytes)
                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        extracted_pages.append({"text": page_text, "page": page_idx + 1})
            except Exception as e:
                print(f"[RAG_WARN] PDF extraction error for {filename}: {e}")

        elif ext in [".docx", ".doc"]:
            try:
                import docx
                doc = docx.Document(file_path_or_bytes)
                full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if full_text.strip():
                    extracted_pages.append({"text": full_text, "page": 1})
            except Exception as e:
                print(f"[RAG_WARN] DOCX extraction error for {filename}: {e}")

        elif ext in [".txt", ".md", ".csv", ".json", ".yaml", ".yml"]:
            try:
                if isinstance(file_path_or_bytes, str):
                    with open(file_path_or_bytes, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                else:
                    text = file_path_or_bytes.read().decode("utf-8", errors="replace")
                if text.strip():
                    extracted_pages.append({"text": text, "page": 1})
            except Exception as e:
                print(f"[RAG_WARN] Text extraction error for {filename}: {e}")

        return extracted_pages

    def chunk_text(self, text: str, source: str, page: Optional[int] = None) -> List[DocumentChunk]:
        """Split text into overlapping character chunks preserving sentences."""
        clean_text = re.sub(r"\s+", " ", text).strip()
        if not clean_text:
            return []

        chunks = []
        start = 0
        chunk_id = len(self.chunks) + 1

        while start < len(clean_text):
            end = min(start + self.chunk_size, len(clean_text))
            # Try to snap end to punctuation if near end
            if end < len(clean_text):
                punct_match = re.search(r"[\.\?\!\n]", clean_text[end-50:end+50])
                if punct_match:
                    end = (end - 50) + punct_match.end()

            chunk_str = clean_text[start:end].strip()
            if len(chunk_str) > 20:
                chunks.append(DocumentChunk(content=chunk_str, source=source, chunk_id=chunk_id, page=page))
                chunk_id += 1

            if end >= len(clean_text):
                break
            start += max(1, self.chunk_size - self.chunk_overlap)

        return chunks

    def add_document(self, file_path_or_bytes, filename: str):
        """Extract and index a document."""
        pages = self.extract_text(file_path_or_bytes, filename)
        new_chunks = []
        for p in pages:
            new_chunks.extend(self.chunk_text(p["text"], source=filename, page=p["page"]))

        self.chunks.extend(new_chunks)
        self._build_index()
        print(f"[RAG_INDEXED] Added '{filename}': {len(new_chunks)} chunks created. Total chunks: {len(self.chunks)}")
        return len(new_chunks)

    def add_directory(self, dir_path: str) -> int:
        """Scan directory and index all supported documents."""
        if not os.path.exists(dir_path):
            return 0
        total_docs = 0
        for root, _, files in os.walk(dir_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in [".pdf", ".docx", ".txt", ".md"]:
                    fpath = os.path.join(root, file)
                    self.add_document(fpath, file)
                    total_docs += 1
        return total_docs

    def _build_index(self):
        """Fit TF-IDF index across all current chunks."""
        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        corpus = [c.content for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most relevant chunks for a user query.
        Returns list of dicts with content, source, page, chunk_id, and similarity_score.
        """
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(sims[idx])
            if score > 0.05:  # Relevance threshold
                chunk = self.chunks[idx]
                results.append({
                    "content": chunk.content,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "score": round(score, 3)
                })

        return results

    def format_citations(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into verifiable source citations."""
        if not results:
            return "No relevant document passages found."

        citations = []
        for i, res in enumerate(results, start=1):
            page_info = f", Page {res['page']}" if res.get("page") else ""
            citations.append(
                f"**[{i}] {res['source']}{page_info}** (Relevance: {res['score']:.0%}):\n"
                f"> \"{res['content']}\""
            )
        return "\n\n".join(citations)
