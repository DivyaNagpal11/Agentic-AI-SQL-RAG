"""
RAG (Retrieval Augmented Generation) agent for handling unstructured data queries.
"""
import logging
from typing import Dict, Any, List, Optional
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.manager import CallbackManager
from langchain.schema import AgentAction, AgentFinish, Document
from langchain.tools.base import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

# Use correct import paths based on project structure
from scripts.agents.primary_agent import BaseAgent, initialize_llm
from scripts.retrieval_strategy import retrieve_from_vector_db
from scripts.config import CHROMA_DB_PATH, TOP_K

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGAgent(BaseAgent):
    """
    Agent for retrieving information from unstructured data using RAG.
    """
    
    def __init__(self, collection_name: str = "documents"):
        super().__init__(
            name="rag_agent",
            description="Retrieves information from documents to answer questions about unstructured data."
        )
        self.collection_name = collection_name
        self.qa_chain = None
        self._setup_tools()
        
    def _setup_tools(self) -> None:
        """Set up the tools for this agent."""
        # Tool for retrieving documents
        retrieve_tool = Tool(
            name="retrieve_documents",
            func=self._retrieve_documents,
            description="Retrieves relevant documents based on a query."
        )
        
        # Tool for answering based on retrieved documents
        answer_tool = Tool(
            name="answer_from_documents",
            func=self._answer_from_documents,
            description="Generates an answer based on the retrieved documents and the original query."
        )
        
        self.add_tool(retrieve_tool)
        self.add_tool(answer_tool)
        
    def _retrieve_documents(self, query: str) -> str:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query: User query string
            
        Returns:
            String representation of the retrieved documents
        """
        logger.info(f"RAG Agent retrieving documents for: '{query}'")
        
        try:
            documents = retrieve_from_vector_db(
                query=query,
                collection_name=self.collection_name,
                top_k=TOP_K
            )
            
            if not documents:
                return "No relevant documents found."
            
            # Format the documents for display
            formatted_docs = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page_number", "N/A")
                formatted_docs.append(f"Document {i+1} [Source: {source}, Page: {page}]:\n{doc.page_content}\n")
            
            return "\n".join(formatted_docs)
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return f"Error retrieving documents: {str(e)}"
    
    def _answer_from_documents(self, input_str: str) -> str:
        """
        Generate an answer based on retrieved documents and query.
        
        Args:
            input_str: String containing both the query and documents (formatted by the agent)
            
        Returns:
            Answer generated from the documents
        """
        try:
            # Parse input to separate query from documents
            # This is a simplified approach - in practice, you'd use a more robust parsing method
            if "Query:" in input_str and "Documents:" in input_str:
                parts = input_str.split("Documents:")
                query = parts[0].replace("Query:", "").strip()
                docs_text = parts[1].strip()
            else:
                query = input_str
                docs_text = self._retrieve_documents(query)
            
            logger.info(f"Generating answer for query: '{query}'")
            
            # Initialize QA chain if not already done
            if not self.qa_chain:
                self._initialize_qa_chain()
            
            # Run the QA chain
            if self.qa_chain:
                result = self.qa_chain.run(query)
                return result
            else:
                # Fallback: use the LLM directly
                prompt = f"""
                Based on the following information, please answer the question.
                
                Question: {query}
                
                Context Information:
                {docs_text}
                
                Answer:
                """
                return self.llm.invoke(prompt).content
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
    
    def _initialize_qa_chain(self) -> None:
        """Initialize the QA chain with the vector store."""
        try:
            # Initialize embedding function
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            except Exception as e:
                logger.error(f"Error initializing HuggingFace embeddings: {str(e)}")
                try:
                    from langchain_community.embeddings import OllamaEmbeddings
                    embedding_function = OllamaEmbeddings(model="llama3")
                except:
                    logger.error("Falling back to fast embeddings")
                    from langchain_community.embeddings import FlagEmbedding
                    embedding_function = FlagEmbedding(model_name="BAAI/bge-small-en")
            
            # Load vector store
            vector_store = Chroma(
                persist_directory=str(CHROMA_DB_PATH),
                embedding_function=embedding_function,
                collection_name=self.collection_name
            )
            
            # Create retriever
            retriever = vector_store.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={"k": TOP_K}
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Other options: map_reduce, refine, map_rerank
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
        except Exception as e:
            logger.error(f"Error initializing QA chain: {str(e)}")
            self.qa_chain = None
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the RAG agent on the given query.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with agent response and metadata
        """
        logger.info(f"Running RAG agent with query: '{query}'")
        
        # First try using the agent executor
        if not self.agent_executor:
            self.build_agent()
        
        try:
            # Try using the agent executor approach first
            if self.agent_executor:
                response = self.agent_executor.run(query)
                return {
                    "response": response,
                    "agent": self.name,
                    "success": True,
                    "error": None
                }
            
            # If no agent executor, fall back to direct document retrieval and answering
            docs = self._retrieve_documents(query)
            response = self._answer_from_documents(f"Query: {query}\nDocuments: {docs}")
            return {
                "response": response,
                "agent": self.name,
                "success": True,
                "error": None,
                "fallback_used": True
            }
        except Exception as e:
            logger.error(f"Error in RAG agent: {str(e)}")
            
            # Final fallback for robustness
            try:
                # One more attempt with direct LLM approach
                docs = self._retrieve_documents(query)
                answer = self._answer_from_documents(f"Query: {query}\nDocuments: {docs}")
                
                return {
                    "response": answer,
                    "agent": self.name,
                    "success": True,
                    "error": None,
                    "fallback_used": True
                }
            except Exception as e2:
                # If all approaches fail, return error
                return {
                    "response": f"I encountered an error while processing your request about document data: {str(e)}",
                    "agent": self.name,
                    "success": False,
                    "error": str(e)
                }

if __name__ == "__main__":
    # Test the RAG agent
    rag_agent = RAGAgent()
    
    test_query = "What information do we have about product features?"
    result = rag_agent.run(test_query)
    
    print(f"Test result: {result}")