import json
import os
from pathlib import Path
from typing import List, Tuple

import dropbox
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

import shutil

VECTOR_INDEX_PATH = "rag_index.faiss"
VECTOR_META_PATH = "rag_metadata.json"
SYNC_META_PATH = "sync_metadata.json"


def sync_dropbox_files(local_folder: str = "knowledge_base") -> List[str]:
    load_dotenv()
    
    access_token = os.getenv("DROPBOX_ACCESS_TOKEN")
    
    if not access_token:
        print("Dropbox access token missing. Skipping sync.")
        return []
        
    updated_files = []
    sync_meta_file = Path(SYNC_META_PATH)
    
    if sync_meta_file.exists():
        try:
            sync_meta = json.loads(sync_meta_file.read_text(encoding="utf-8"))
        except Exception:
            sync_meta = {}
    else:
        sync_meta = {}
        
    try:
        dbx = dropbox.Dropbox(access_token)
        
        folder = Path(local_folder)
        folder.mkdir(parents=True, exist_ok=True)
        
        results = dbx.files_list_folder('/Serivce manual')
        
        while True:
            for entry in results.entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.lower().endswith('.pdf'):
                    local_path = folder / entry.name
                    server_modified_str = str(entry.server_modified)
                    
                    # Check if file is new or modified
                    if entry.name not in sync_meta or sync_meta[entry.name] != server_modified_str or not local_path.exists():
                        try:
                            print(f"Downloading file from Dropbox: {entry.name.encode('utf-8', 'ignore').decode('utf-8')}")
                        except Exception:
                            pass
                        dbx.files_download_to_file(str(local_path), entry.path_lower)
                        sync_meta[entry.name] = server_modified_str
                        updated_files.append(str(local_path))
            
            if not results.has_more:
                break
            results = dbx.files_list_folder_continue(results.cursor)
                    
        # Save updated sync metadata
        sync_meta_file.write_text(json.dumps(sync_meta, ensure_ascii=False, indent=4), encoding="utf-8")
        
    except Exception as e:
        print(f"Error syncing with Dropbox: {e}")
        
    return updated_files


def _build_embeddings_client() -> OpenAIEmbeddings:
    load_dotenv()

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise ValueError("API_KEY is missing in .env")
    if not base_url:
        raise ValueError("BASE_URL is missing in .env")

    try:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
            base_url=base_url,
        )
    except TypeError:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
            openai_api_base=base_url,
        )


def load_and_process_documents(folder_path: str, specific_files: List[str] = None) -> Tuple[List[str], int]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return [], 0

    if specific_files:
        pdf_files = [Path(f) for f in specific_files if Path(f).exists() and Path(f).name.lower().endswith('.pdf')]
    else:
        pdf_files = sorted(folder.glob("*.pdf"))
        
    if not pdf_files:
        return [], 0

    raw_sections: List[str] = []
    for pdf_file in pdf_files:
        try:
            reader = PdfReader(str(pdf_file))
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    raw_sections.append(f"[{pdf_file.name} - page {page_num}]\n{page_text}")
        except Exception as e:
            print(f"Skipped {pdf_file.name} due to PDF reading error: {e}")
            continue

    if not raw_sections:
        return [], 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks: List[str] = []
    total_chars = 0
    for section in raw_sections:
        total_chars += len(section)
        split_chunks = splitter.split_text(section)
        chunks.extend(chunk for chunk in split_chunks if chunk.strip())

    return chunks, total_chars


def create_or_update_vector_database(
    folder_path: str,
    index_path: str = VECTOR_INDEX_PATH,
    metadata_path: str = VECTOR_META_PATH,
    new_files: List[str] = None
) -> Tuple[int, int]:
    
    pdf_count = len(list(Path(folder_path).glob("*.pdf")))
    
    # Load existing metadata to append
    metadata_file = Path(metadata_path)
    existing_chunks = []
    total_chars = 0
    
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            existing_chunks = metadata.get("chunks", [])
            total_chars = metadata.get("total_chars", 0)
        except Exception:
            pass

    index_exists = os.path.exists(index_path)
    meta_exists = metadata_file.exists()

    # If full rebuild is required (no index exists or new_files is None)
    if new_files is None or not index_exists or not meta_exists:
        if index_exists:
            os.remove(index_path)
        if metadata_file.exists():
            os.remove(metadata_file)
            
        print("Rebuilding entire FAISS index...")
        existing_chunks = []
        total_chars = 0
        chunks_to_embed, new_chars = load_and_process_documents(folder_path)
        
    else:
        # Index already exists and no files downloaded this sync run
        if not new_files:
            return pdf_count, total_chars
            
        # Partial Indexing (Incremental)
        print(f"Appending {len(new_files)} new files to the FAISS index...")
        chunks_to_embed, new_chars = load_and_process_documents(folder_path, specific_files=new_files)

    if chunks_to_embed:
        embeddings = _build_embeddings_client()
        vectors = embeddings.embed_documents(chunks_to_embed)
        
        if vectors:
            vector_array = np.array(vectors, dtype="float32")
            
            if os.path.exists(index_path) and new_files is not None:
                # Append to existing index
                index = faiss.read_index(index_path)
                index.add(vector_array)
            else:
                # Create brand new index
                index = faiss.IndexFlatL2(vector_array.shape[1])
                index.add(vector_array)

            faiss.write_index(index, index_path)
            
            existing_chunks.extend(chunks_to_embed)
            total_chars += new_chars
    
    # Always write the updated metadata counts if we completed the logic
    # (even if chunks_to_embed was empty this pass)
    if existing_chunks or chunks_to_embed:
        metadata_file.write_text(
            json.dumps({
                "chunks": existing_chunks or chunks_to_embed, 
                "pdf_count": pdf_count, 
                "total_chars": total_chars
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        
    return pdf_count, total_chars


def _load_vector_database(
    index_path: str = VECTOR_INDEX_PATH,
    metadata_path: str = VECTOR_META_PATH,
) -> Tuple[faiss.Index | None, List[str]]:
    index_file = Path(index_path)
    metadata_file = Path(metadata_path)

    if not index_file.exists() or not metadata_file.exists():
        return None, []

    try:
        index = faiss.read_index(str(index_file))
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        chunks = metadata.get("chunks", [])
    except Exception:
        return None, []

    if not chunks or index.ntotal == 0:
        return None, []

    return index, chunks


def similarity_search(
    query: str,
    k: int = 8,
    knowledge_folder: str = "knowledge_base",
    index_path: str = VECTOR_INDEX_PATH,
    metadata_path: str = VECTOR_META_PATH,
) -> List[str]:
    if not query.strip():
        return []

    # Increase 'k' (context window) significantly if the user asks about the uploaded files
    meta_keywords = ["ملف", "ملفات", "قائمة", "ماذا تعرف", "كتالوج", "الكتالوجات", "تعرف"]
    if any(keyword in query for keyword in meta_keywords):
        k = 15

    try:
        index, chunks = _load_vector_database(index_path=index_path, metadata_path=metadata_path)

        if index is None or not chunks:
            created_stats = create_or_update_vector_database(
                folder_path=knowledge_folder,
                index_path=index_path,
                metadata_path=metadata_path,
            )
            if created_stats[0] == 0:
                return []

            index, chunks = _load_vector_database(index_path=index_path, metadata_path=metadata_path)
            if index is None or not chunks:
                return []

        embeddings = _build_embeddings_client()
        query_vector = np.array([embeddings.embed_query(query)], dtype="float32")

        top_k = max(1, min(k, index.ntotal))
        _, indices = index.search(query_vector, top_k)

        results: List[str] = []
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                results.append(chunks[idx])
        return results
    except Exception:
        return []

