import os
import asyncio

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
    def __init__(self, **kwargs) :
        # The default node parser
        self.node_parser = self.node_parser_factory(
            type = kwargs.get("node_parser", "sentence"),
        )
        
        # The default embedding model
        self.embed_model = self.embed_model_factory(
            type = kwargs.get("embed_model", "fastembed"),
        )
        
        # The default base llm and llm used for metadata generation
        self.base_llm = self.llm_models(type="groq")
        self.metadata_llm = self.llm_models(type="gemini-flash")
        
        # By default, set to all the metadata extractors
        self.extractors = self.metadata_extractors(
            extractor_types = kwargs.get(
                "extractors",
                ["title", "summary", "questions-answered", "keyword"]
            ),
        )
    
    
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
    
    
    def metadata_extractors(self, extractor_types, **kwargs) :
        extractors = {
            "title": TitleExtractor(
                nodes=kwargs.get("title_nodes", 5),
                llm=kwargs.get("llm", self.metadata_llm),
            ),
            
            "summary": SummaryExtractor(
                summaries=kwargs.get("summaries", ["prev", "self", "next"]),
                llm=kwargs.get("llm", self.metadata_llm),
            ),
            
            "questions-answered": QuestionsAnsweredExtractor(
                questions=kwargs.get("questions", 5),
                llm=kwargs.get("llm", self.metadata_llm),
            ),
            
            "keyword": KeywordExtractor(
                keywords=kwargs.get("keywords", 5),
                llm=kwargs.get("llm", self.metadata_llm),
            ),
        }
        
        
        return [extractors[type] for type in extractor_types]
    


class RedisStore :
    def __init__(self, index_name, index_prefix) :
        # Initialize the vector store
        self.vector_store = RedisVectorStore(
            schema=self.get_custom_schema(index_name, index_prefix),
            redis_url="redis://localhost:6379",
        )
        
        # Set up the ingestion cache layer
        self.cache = IngestionCache(
            cache=RedisCache.from_host_and_port("localhost", 6379),
            collection=f"redis_cache_{index_name}",
        )
        
        # Initialize the document store
        self.docstore = RedisDocumentStore.from_host_and_port(
            "localhost",
            6379,
            namespace=f"document_store_{index_name}"
        )
    
    
    def get_custom_schema(
            self, 
            index_name = "rough_index", 
            index_prefix = "doc"
        ) :
        """Return the custom schema for the index
        """
        custom_schema = IndexSchema.from_dict(
            {
                "index": {"name": index_name, "prefix": index_prefix},
                # customize fields that are indexed
                "fields": [
                    # required fields for llamaindex
                    {"type": "tag", "name": "id"},
                    {"type": "tag", "name": "doc_id"},
                    {"type": "text", "name": "text"},
                    # custom vector field for bge-small-en-v1.5 embeddings
                    {
                        "type": "vector",
                        "name": "vector",
                        "attrs": {
                            "dims": 384,
                            "algorithm": "hnsw",
                            "distance_metric": "cosine",
                        },
                    },
                ],
            }
        )
        
        return custom_schema
        

    
class DocumentIngestionPipeline :
    def __init__(self, rag_config, store) :
        self.config = rag_config
        self.store = store
        

    def generate_parent_child_nodes(self, base_nodes) :
        sub_chunk_sizes = [128, 256, 512]
        sub_node_parsers = [
            self.config.node_parser_factory(
                "sentence", 
                chunk_size=c, 
                chunk_overlap=20
            ) for c in sub_chunk_sizes
        ]

        all_nodes = []
        for base_node in base_nodes:
            for n in sub_node_parsers:
                sub_nodes = n.get_nodes_from_documents([base_node])
                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                all_nodes.extend(sub_inodes)

            # also add original node to node
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(original_node)

        return all_nodes
    
    def p_c_retrieval_w_metadata(self, docs) :
        """Parent-Child retrieval with metadata
        """
        
        # Inititalize the node parser and create base nodes
        node_parser = self.config.node_parser # SentenceSplitter(chunk_size=1024)
        base_nodes = node_parser.get_nodes_from_documents(docs)
        
        # Generate parent-child nodes
        all_nodes = self.generate_parent_child_nodes(base_nodes)
        
        all_nodes_dict = {n.node_id: n for n in all_nodes}
        
        pipeline = IngestionPipeline(
            transformations=[
                *self.config.extractors,
                self.config.embed_model
            ],
            docstore=self.store.docstore,
            vector_store=self.store.vector_store,
            cache=self.store.cache,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )
        
        return pipeline, all_nodes_dict
    
    
    def p_c_retrieval_wo_metadata(self, docs) :
        """Parent Child retrieval without metadata
        """
        
        # Inititalize the node parser and create base nodes
        node_parser = self.config.node_parser # SentenceSplitter(chunk_size=1024)
        base_nodes = node_parser.get_nodes_from_documents(docs)
        
        # Generate parent-child nodes
        all_nodes = self.generate_parent_child_nodes(base_nodes)
        
        all_nodes_dict = {n.node_id: n for n in all_nodes}
        
        pipeline = IngestionPipeline(
            transformations=[
                self.config.embed_model
            ],
            docstore=self.store.docstore,
            vector_store=self.store.vector_store,
            cache=self.store.cache,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )
        
        return pipeline, all_nodes_dict
    
    
    def nodes_pipeline_run_async(self, pipeline, all_nodes) :
        i = 0
        total_len = len(all_nodes)
        loop = asyncio.get_event_loop()

        print("STARTED...")
        while i < total_len :
            print(f"At node count(i): {i}")
            print(f"Percent: {(i/total_len)*100}")
            try :
                if i+1000 < total_len :
                    nodes = loop.run_until_complete(pipeline.arun(nodes=all_nodes[i: i+1000],
                                                                show_progress=True,))
                else :
                    nodes = loop.run_until_complete(pipeline.arun(nodes=all_nodes[i:],
                                                                show_progress=True,))
            except Exception as ex :
                print(f"*"*100)
                print(f"Exception noted: {ex}")
                continue

            print(f"Ingested {len(nodes)} Nodes")
            i += 1000

        print("DONE..")
        
        return pipeline