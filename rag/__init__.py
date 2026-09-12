import time
import uuid
import nltk
import streamlit as st
import os
import shutil
import hashlib
import pickle

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

from bs4 import BeautifulSoup
import markdownify
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy 
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.storage import InMemoryStore 
from langchain_community.retrievers import BM25Retriever 
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

from selenium import webdriver
from selenium.webdriver.firefox.options import Options

# Standard Desktop User-Agent string for browser options
CUSTOM_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

@st.cache_resource
def get_flashrank_reranker(top_n=5):
    return FlashrankRerank(top_n=top_n)


class RAG:
    
    def __init__(self, url: str, google_api_key: str, write_function=None):
        if write_function is None:
            self.write_function = print
        else:
            self.write_function = write_function

        self.url = url
        self.google_api_key = google_api_key
        self.cache_dir = "./vector_cache"
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        try:
            self.retriever = self.read_website() 
        except Exception as e:
            self.write_function(f"Failed to read website: {e}")
            raise

        self.write_function("Initializing LLM...")
        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=self.google_api_key)

        prompt = ChatPromptTemplate.from_template("""
        Act as a chatbot and answer questions about the website. Refer to the following context whenever possible.
        <context>
        {context}
        </context>
        ---
        Question: {input}                                                                                                                                                                                                                                                                                                                                                                                                                
        """)

        self.write_function("Creating document chain...")
        self.document_chain = create_stuff_documents_chain(llm, prompt)

    def _cleanup_old_caches(self, max_sites=5):
        """Garbage Collector: Keeps only the 5 most recently saved websites."""
        site_folders = [os.path.join(self.cache_dir, d) for d in os.listdir(self.cache_dir) if os.path.isdir(os.path.join(self.cache_dir, d))]
        
        if len(site_folders) > max_sites:
            self.write_function("Running Garbage Collector: Removing oldest caches...")
            site_folders.sort(key=os.path.getmtime) 
            while len(site_folders) > max_sites:
                folder_to_delete = site_folders.pop(0)
                shutil.rmtree(folder_to_delete)

    def read_website(self):
        url_hash = hashlib.md5(self.url.encode()).hexdigest()
        site_cache_path = os.path.join(self.cache_dir, url_hash)

        embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2", 
            google_api_key=self.google_api_key
        )

        child_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        id_key = "doc_id"
        docstore = InMemoryStore()

        # --- CACHE CHECK ---
        if os.path.exists(site_cache_path):
            self.write_function("⚡ Local Cache Found! Skipping scraper and rate limits...")
            self.write_function("Loading Vector Database from disk...")
            
            db = FAISS.load_local(site_cache_path, embedder, allow_dangerous_deserialization=True)
            
            with open(os.path.join(site_cache_path, "parents.pkl"), "rb") as f:
                parent_docs = pickle.load(f)
            
            parent_doc_dict = {doc.metadata[id_key]: doc for doc in parent_docs}
            docstore.mset(list(parent_doc_dict.items()))
            
        else:
            # --- SCRAPE & BUILD ---
            self.write_function("No cache found. Booting Web Scraper...")
            options = Options()
            options.add_argument('-headless')  
            options.add_argument('--disable-gpu') 
            options.add_argument('--no-sandbox') 
            options.add_argument('--disable-dev-shm-usage') 
            # Inject real browser User-Agent
            options.add_argument(f"user-agent={CUSTOM_USER_AGENT}")
            
            driver = webdriver.Firefox(options=options)
            driver.get(self.url)
            text_html = driver.page_source
            driver.quit()

            self.write_function("Converting HTML to Structured Markdown...")
            soup = BeautifulSoup(text_html, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            core_content = soup.find('main') or soup.find('article') or soup.find('body') or soup
            md_text = markdownify.markdownify(str(core_content), heading_style="ATX")

            self.write_function("Executing Hierarchical Parent-Child Chunking...")
            headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            header_splits = markdown_splitter.split_text(md_text)

            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            parent_docs = parent_splitter.split_documents(header_splits)

            if not parent_docs:
                raise ValueError("No extractable content found.")

            parent_doc_dict = {}
            for doc in parent_docs:
                doc_id = str(uuid.uuid4())
                doc.metadata[id_key] = doc_id
                parent_doc_dict[doc_id] = doc
            
            docstore.mset(list(parent_doc_dict.items()))

            child_docs = []
            for doc in parent_docs:
                _id = doc.metadata[id_key]
                _sub_docs = child_splitter.split_documents([doc])
                for _doc in _sub_docs:
                    _doc.metadata[id_key] = _id  
                child_docs.extend(_sub_docs)
            
            self.write_function(f"Embedding {len(child_docs)} chunks via Google API...")
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
                            self.write_function(f"⚠️ Rate Limit hit. Waiting {sleep_time}s...")
                            time.sleep(sleep_time)
                            retry_count += 1
                        else:
                            raise e 
                
                if i + batch_size < len(child_docs):
                    self.write_function("Pausing 60 seconds to clear Google API limits...")
                    time.sleep(60)

            # Save the vectors and text permanently
            self.write_function("Saving Vector Database to Local Cache...")
            db.save_local(site_cache_path)
            with open(os.path.join(site_cache_path, "parents.pkl"), "wb") as f:
                pickle.dump(parent_docs, f)
            
            self._cleanup_old_caches()

        # --- HYBRID SEARCH & RERANKER ---
        self.write_function("Fusing Semantic Search and BM25...")
        bm25_retriever = BM25Retriever.from_documents(parent_docs)
        bm25_retriever.k = 7

        parent_retriever = ParentDocumentRetriever(
            vectorstore=db,
            docstore=docstore,
            child_splitter=child_splitter,
            id_key=id_key,
            search_kwargs={"k": 10} 
        )

        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, parent_retriever],
            weights=[0.4, 0.6] 
        )

        self.write_function("Wrapping with FlashRank Cross-Encoder...")
        compressor = get_flashrank_reranker(top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=hybrid_retriever
        )

        self.write_function("System Ready!")
        return compression_retriever
      
    def get_response(self, question: str):
        if not getattr(self, 'retriever', None):
            return {"answer": "No readable text found.", "sources": []}

        retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)
        result = retrieval_chain.invoke({"input": question})
        
        sources = [doc.page_content for doc in result['context']]
        
        return {
            "answer": result['answer'],
            "sources": sources
        }