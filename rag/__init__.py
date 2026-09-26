"""
RAG-CRAW Core Pipeline: Hardened Enterprise-Grade Ingestion, Hybrid Search & Reranking.
Designed for high security, resilient scraping, zero unsafe deserialization, and deterministic caching.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import urllib.parse
import uuid

from bs4 import BeautifulSoup
import faiss
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    ParentDocumentRetriever,
)
from langchain.storage import InMemoryStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import markdownify
import nltk
import requests
from selenium import webdriver
import streamlit as st

# Ensure Windows terminal handles UTF-8 correctly
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Ensure required NLTK resources exist
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

# Standard desktop User-Agent to avoid generic scraping blocks
CUSTOM_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Standard tracking parameters to strip during URL normalization
TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "ref",
    "source",
    "token",
    "_ga",
    "_gl",
    "mc_cid",
    "mc_eid",
}


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Safely print to stdout without crashing on consoles with limited character encodings."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"), **kwargs)


def normalize_url(raw_url: str) -> str:
    """
    Cleans and standardizes a URL before cache key generation:
    - Strips whitespace
    - Defaults missing scheme to https://
    - Normalizes scheme and host to lowercase
    - Strips trailing slash on root paths
    - Filters out tracking/marketing query parameters (utm_*, fbclid, etc.)
    - Removes URL fragment anchors
    - Sorts remaining query parameters deterministically
    """
    clean_url = raw_url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    parsed = urllib.parse.urlparse(clean_url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path

    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")

    # Filter out tracking query parameters
    query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    sanitized_params = [
        (k, v)
        for k, v in query_params
        if not (k.lower().startswith("utm_") or k.lower() in TRACKING_PARAMS)
    ]
    sanitized_params.sort(key=lambda item: item[0])
    normalized_query = urllib.parse.urlencode(sanitized_params)

    # Discard fragments (#...) to avoid duplicate indices for anchor links
    return urllib.parse.urlunparse((scheme, netloc, path, parsed.params, normalized_query, ""))


def save_faiss_safely(db: FAISS, folder_path: str) -> None:
    """
    Persists FAISS vector index using native binary write and serializes
    the associated in-memory docstore as JSON, entirely bypassing raw pickle.
    """
    os.makedirs(folder_path, exist_ok=True)
    # 1. Native FAISS binary index file
    faiss.write_index(db.index, os.path.join(folder_path, "index.faiss"))

    # 2. JSON-serialize child document store
    docstore_dict = {
        doc_id: {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc_id, doc in db.docstore._dict.items()
    }

    # 3. JSON-serialize index_to_docstore_id mapping (with integer keys preserved)
    metadata_payload = {
        "docstore": docstore_dict,
        "index_to_docstore_id": db.index_to_docstore_id,
    }
    with open(os.path.join(folder_path, "docstore.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)


def load_faiss_safely(folder_path: str, embedder: GoogleGenerativeAIEmbeddings) -> FAISS:
    """
    Reconstructs FAISS vector store using native binary reader and JSON docstore,
    with zero unsafe pickle deserialization.
    """
    index_file = os.path.join(folder_path, "index.faiss")
    docstore_file = os.path.join(folder_path, "docstore.json")

    if not os.path.exists(index_file) or not os.path.exists(docstore_file):
        raise FileNotFoundError(f"Safe FAISS files missing in {folder_path}")

    index = faiss.read_index(index_file)
    with open(docstore_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    docstore = InMemoryDocstore({
        k: Document(page_content=v["page_content"], metadata=v.get("metadata", {}))
        for k, v in data["docstore"].items()
    })
    index_to_docstore_id = {int(k): v for k, v in data["index_to_docstore_id"].items()}

    return FAISS(
        embedding_function=embedder,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def save_parent_docs_json(parent_docs: List[Document], folder_path: str) -> None:
    """Serializes parent document contents and metadata to a clean JSON file."""
    os.makedirs(folder_path, exist_ok=True)
    serialized = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in parent_docs
    ]
    with open(os.path.join(folder_path, "parents.json"), "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)


def load_parent_docs_json(folder_path: str) -> List[Document]:
    """Deserializes parent documents safely from parents.json."""
    parents_file = os.path.join(folder_path, "parents.json")
    if not os.path.exists(parents_file):
        raise FileNotFoundError(f"parents.json not found in {folder_path}")
    with open(parents_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in data
    ]


def save_metadata(folder_path: str, meta: Dict[str, Any]) -> None:
    """Writes provenance metadata for cache validation and freshness checks."""
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_metadata(folder_path: str) -> Optional[Dict[str, Any]]:
    """Loads cache provenance metadata if present."""
    meta_path = os.path.join(folder_path, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@st.cache_resource
def get_flashrank_reranker(top_n: int = 5) -> FlashrankRerank:
    """Singleton cached instance of FlashRank cross-encoder."""
    return FlashrankRerank(top_n=top_n)


class RAG:
    """
    Production-hardened Retrieval-Augmented Generation pipeline.
    Combines hierarchical chunking, hybrid BM25 + FAISS search,
    FlashRank cross-encoder reranking, and secure JSON/native-binary caching.
    """

    def __init__(
        self,
        url: str,
        google_api_key: str,
        write_function: Optional[Callable[[str], None]] = None,
        cache_dir: str = "./vector_cache",
        **kwargs: Any,
    ) -> None:
        if write_function is None:
            self.write_function = safe_print
        else:
            def wrapped_write(msg: str) -> None:
                try:
                    write_function(msg)
                except UnicodeEncodeError:
                    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
                    safe_msg = str(msg).encode(enc, errors="replace").decode(enc, errors="replace")
                    write_function(safe_msg)
            self.write_function = wrapped_write

        # URL normalization to produce deterministic, tracking-free cache keys
        self.raw_url = url.strip()
        self.normalized_url = normalize_url(self.raw_url)
        self.google_api_key = google_api_key
        self.cache_dir = cache_dir

        # Scoped document store instance for parent documents
        self.docstore = InMemoryStore()
        self.id_key = "doc_id"
        self.metadata: Optional[Dict[str, Any]] = None

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        try:
            self.retriever = self.read_website()
        except Exception as e:
            self.write_function(f"Failed to read website: {e}")
            raise

        self.write_function("Initializing LLM...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            google_api_key=self.google_api_key,
        )

        prompt = ChatPromptTemplate.from_template("""
        Act as a helpful chatbot and answer questions about the website. Refer to the following context whenever possible.
        <context>
        {context}
        </context>
        ---
        Question: {input}
        """)

        self.write_function("Creating document chain...")
        self.document_chain = create_stuff_documents_chain(llm, prompt)

    def _cleanup_old_caches(self, max_sites: int = 5) -> None:
        """LRU Garbage Collector: Keeps only the N most recently modified caches."""
        site_folders = [
            os.path.join(self.cache_dir, d)
            for d in os.listdir(self.cache_dir)
            if os.path.isdir(os.path.join(self.cache_dir, d))
        ]

        if len(site_folders) > max_sites:
            self.write_function("Running Garbage Collector: Pruning oldest caches...")
            site_folders.sort(key=os.path.getmtime)
            while len(site_folders) > max_sites:
                folder_to_delete = site_folders.pop(0)
                shutil.rmtree(folder_to_delete, ignore_errors=True)

    def _scrape_html(self) -> str:
        """
        Multi-tier scraper:
        1. Fast HTTP request using desktop headers
        2. Fallback to Headless Browser (Chrome -> Edge -> Firefox) with
           strict try...finally lifecycle management and anti-detection flags.
        """
        # Tier 1: Fast HTTP GET with requests
        try:
            self.write_function("Fetching webpage content via direct HTTP request...")
            headers = {
                "User-Agent": CUSTOM_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = requests.get(self.normalized_url, headers=headers, timeout=15)
            if resp.status_code == 200 and len(resp.text) > 300:
                soup_test = BeautifulSoup(resp.text, "html.parser")
                body_text = soup_test.get_text(strip=True)
                if len(body_text) > 150:
                    self.write_function(f"Direct fetch successful ({len(resp.text)} bytes).")
                    return resp.text
        except Exception as req_err:
            self.write_function(f"Direct fetch bypassed ({req_err}). Launching headless browser...")

        # Tier 2: Headless Browser via Selenium
        self.write_function("Booting headless browser for dynamic JavaScript rendering...")
        driver = None
        browser_errors = []

        # 1. Attempt Chrome
        try:
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            c_opts = ChromeOptions()
            c_opts.add_argument("--headless=new")
            c_opts.add_argument("--disable-gpu")
            c_opts.add_argument("--no-sandbox")
            c_opts.add_argument("--disable-dev-shm-usage")
            c_opts.add_argument(f"user-agent={CUSTOM_USER_AGENT}")
            c_opts.add_argument("--disable-blink-features=AutomationControlled")
            c_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
            c_opts.add_experimental_option("useAutomationExtension", False)

            driver = webdriver.Chrome(options=c_opts)
            try:
                driver.execute_cdp_cmd(
                    "Page.addScriptToEvaluateOnNewDocument",
                    {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
                )
            except Exception:
                pass
        except Exception as e:
            browser_errors.append(f"Chrome: {e}")

        # 2. Attempt Edge
        if driver is None:
            try:
                from selenium.webdriver.edge.options import Options as EdgeOptions
                e_opts = EdgeOptions()
                e_opts.add_argument("--headless=new")
                e_opts.add_argument("--disable-gpu")
                e_opts.add_argument("--no-sandbox")
                e_opts.add_argument("--disable-dev-shm-usage")
                e_opts.add_argument(f"user-agent={CUSTOM_USER_AGENT}")
                e_opts.add_argument("--disable-blink-features=AutomationControlled")
                e_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
                e_opts.add_experimental_option("useAutomationExtension", False)

                driver = webdriver.Edge(options=e_opts)
            except Exception as e:
                browser_errors.append(f"Edge: {e}")

        # 3. Attempt Firefox
        if driver is None:
            try:
                from selenium.webdriver.firefox.options import Options as FirefoxOptions
                f_opts = FirefoxOptions()
                f_opts.add_argument("-headless")
                f_opts.add_argument("--disable-gpu")
                f_opts.add_argument("--no-sandbox")
                f_opts.add_argument("--disable-dev-shm-usage")
                f_opts.add_argument(f"user-agent={CUSTOM_USER_AGENT}")

                driver = webdriver.Firefox(options=f_opts)
            except Exception as e:
                browser_errors.append(f"Firefox: {e}")

        # Strict browser lifecycle guarantees: guaranteed driver.quit() via try...finally
        if driver is not None:
            try:
                driver.set_page_load_timeout(15)
                driver.set_script_timeout(15)
                driver.get(self.normalized_url)
                time.sleep(2)
                html = driver.page_source
                return html
            finally:
                driver.quit()
        else:
            raise RuntimeError(
                f"Could not load website with any available browser engine. Errors: {'; '.join(browser_errors)}"
            )

    def read_website(self) -> ContextualCompressionRetriever:
        """
        Parses, indexes, and compiles hybrid retriever with safe cache restoration:
        - Cache hit: Loads native FAISS index + parents.json, re-indexes BM25Retriever.
        - Cache miss: Scrapes HTML, chunks hierarchically, embeds child vectors,
          and writes index.faiss + docstore.json + parents.json + metadata.json.
        """
        url_hash = hashlib.md5(self.normalized_url.encode("utf-8")).hexdigest()
        site_cache_path = os.path.join(self.cache_dir, url_hash)

        embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=self.google_api_key,
        )

        child_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

        # Cache validation: Must have valid index.faiss, docstore.json, and parents.json
        has_secure_cache = (
            os.path.exists(site_cache_path)
            and os.path.exists(os.path.join(site_cache_path, "index.faiss"))
            and os.path.exists(os.path.join(site_cache_path, "docstore.json"))
            and os.path.exists(os.path.join(site_cache_path, "parents.json"))
        )

        if has_secure_cache:
            self.write_function("⚡ Secure Cache Hit! Loading Vector Database and Parents from disk...")
            db = load_faiss_safely(site_cache_path, embedder)
            parent_docs = load_parent_docs_json(site_cache_path)
            self.metadata = load_metadata(site_cache_path)

            # Re-populate the scoped InMemoryStore
            parent_doc_dict = {doc.metadata[self.id_key]: doc for doc in parent_docs}
            self.docstore.mset(list(parent_doc_dict.items()))
            self.write_function(f"Loaded {len(parent_docs)} parent documents into store.")

        else:
            self.write_function("No secure cache found. Ingesting website...")
            text_html = self._scrape_html()

            self.write_function("Converting HTML to Structured Markdown...")
            soup = BeautifulSoup(text_html, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            core_content = soup.find("main") or soup.find("article") or soup.find("body") or soup
            md_text = markdownify.markdownify(str(core_content), heading_style="ATX")

            content_hash = hashlib.sha256(md_text.encode("utf-8")).hexdigest()

            self.write_function("Executing Hierarchical Parent-Child Chunking...")
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            header_splits = markdown_splitter.split_text(md_text)

            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            parent_docs = parent_splitter.split_documents(header_splits)

            if not parent_docs:
                raise ValueError("No extractable content found on this website.")

            parent_doc_dict = {}
            for doc in parent_docs:
                doc_id = str(uuid.uuid4())
                doc.metadata[self.id_key] = doc_id
                parent_doc_dict[doc_id] = doc

            self.docstore.mset(list(parent_doc_dict.items()))

            # Generate child chunks mapped to parent UUID
            child_docs: List[Document] = []
            for doc in parent_docs:
                _id = doc.metadata[self.id_key]
                _sub_docs = child_splitter.split_documents([doc])
                for _doc in _sub_docs:
                    _doc.metadata[self.id_key] = _id
                child_docs.extend(_sub_docs)

            self.write_function(f"Embedding {len(child_docs)} child chunks via Gemini Embedding...")
            db = None
            batch_size = 90

            for i in range(0, len(child_docs), batch_size):
                batch = child_docs[i : i + batch_size]
                total_batches = (len(child_docs) + batch_size - 1) // batch_size
                self.write_function(f"Embedding batch {i // batch_size + 1} of {total_batches}...")

                retry_count = 0
                while True:
                    try:
                        if db is None:
                            db = FAISS.from_documents(batch, embedder, distance_strategy=DistanceStrategy.COSINE)
                        else:
                            db.add_documents(batch)
                        break
                    except Exception as e:
                        if "429" in str(e) or "Quota" in str(e):
                            sleep_time = 45 + (retry_count * 15)
                            self.write_function(f"⚠️ Rate limit hit. Backing off for {sleep_time}s...")
                            time.sleep(sleep_time)
                            retry_count += 1
                        else:
                            raise e

                if i + batch_size < len(child_docs):
                    time.sleep(1)

            # Persist FAISS index, docstore, parents, and metadata safely
            self.write_function("Persisting native FAISS index and parents.json...")
            save_faiss_safely(db, site_cache_path)
            save_parent_docs_json(parent_docs, site_cache_path)

            self.metadata = {
                "original_url": self.raw_url,
                "normalized_url": self.normalized_url,
                "content_hash": content_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "parent_chunks_count": len(parent_docs),
                "child_chunks_count": len(child_docs),
                "embedding_model": "models/gemini-embedding-2",
                "llm_model": "gemini-3.1-flash-lite",
            }
            save_metadata(site_cache_path, self.metadata)
            self._cleanup_old_caches()

        # Hybrid Search Assembly: Re-index BM25 immediately over parent_docs
        self.write_function("Fusing Semantic Search and BM25...")
        bm25_retriever = BM25Retriever.from_documents(parent_docs)
        bm25_retriever.k = min(7, len(parent_docs))

        parent_retriever = ParentDocumentRetriever(
            vectorstore=db,
            docstore=self.docstore,
            child_splitter=child_splitter,
            id_key=self.id_key,
            search_kwargs={"k": 10},
        )

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, parent_retriever],
            weights=[0.4, 0.6],
        )

        self.write_function("Wrapping with FlashRank Cross-Encoder...")
        compressor = get_flashrank_reranker(top_n=min(5, len(parent_docs)))
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=hybrid_retriever,
        )

        self.write_function("System Ready!")
        return compression_retriever

    def get_response(self, question: str) -> Dict[str, Any]:
        """
        Executes query against the hybrid compression retrieval chain
        and returns formatted answer and context sources.
        """
        if not getattr(self, "retriever", None):
            return {"answer": "No readable text found.", "sources": []}

        retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)
        result = retrieval_chain.invoke({"input": question})

        sources = [doc.page_content for doc in result.get("context", [])]

        return {
            "answer": result.get("answer", ""),
            "sources": sources,
            "metadata": self.metadata or {},
        }