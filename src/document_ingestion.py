import os

from llama_index.core.schema import MetadataMode, IndexNode
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama

from llama_index.core import Settings
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import (
    SentenceWindowNodeParser,
    SimpleNodeParser,
    SentenceSplitter,
)
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

from llama_index.readers.file import PyMuPDFReader

from llama_index.core import SimpleDirectoryReader
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)

from llama_index.core import StorageContext, load_index_from_storage

from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema

from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")


class RagConfig :
    def __init__(self) :
        self.node_parser = self.node_parser_factory(
            type = "sentence",
            chunk_size = 1024,
        )
        
        self.embed_model = self.embed_model_factory(
            type = "fastembed",
        )
        
        self.base_llm = self.llm_models(type="groq")
    
    
    def node_parser_factory(self, model_type, **kwargs) :
        node_parsers = {
            "sentence-window": SentenceWindowNodeParser.from_defaults(
                window_size=kwargs.get("window_size", 6),
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            ),
            
            "simple": SimpleNodeParser(),
            
            "sentence": SentenceSplitter(
                chunk_size=kwargs.get("chunk_size", 1024),
                chunk_overlap=kwargs.get("chunk_overlap", 64)
            ),
        }
        
        return node_parsers[model_type]
    
    
    def embed_model_factory(self, model_type, **kwargs) :
        embed_models = {
            "huggingface": HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en",
                **kwargs
            ),
            
            "fastembed": FastEmbedEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                *kwargs
            )
        }
        
        return embed_models[model_type]
    
    
    def llm_models(self, type, **kwargs) :
        llm_models = {
            "openai": OpenAI(
                model="gpt-3.5-turbo", 
                temperature=0,
                **kwargs,
            ),
            
            "gemini-flash": Gemini(
                model="models/gemini-1.5-flash", 
                temperature=kwargs.get("temperature", 1),
                request_timeout=kwargs.get("request_timeout", 60),
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
                ],
                **kwargs,
            ),
            
            "gemini-pro": Gemini(
                model="models/gemini-pro", 
                temperature=kwargs.get("temperature", 0), 
                max_tokens=kwargs.get("max_tokens", 512),
                **kwargs
            ),
            
            "groq": Groq(
                model="gemma2-9b-it",
                temperature=kwargs.get("temperature", 0),
                max_tokens=kwargs.get("max_tokens", 512),
                **kwargs,
            ),
            
            "ollama": Ollama(
                model="gemma:2b",
                temperature=kwargs.get("temperature", 0),
                request_timeout=kwargs.get("request_timeout", 60),
                **kwargs,
            )
        }
        
        return llm_models[type]