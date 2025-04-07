"""
Main application script for the Q&A chatbot with Gradio interface.
"""
import os
import logging
import gradio as gr
from typing import Dict, Any, List
import pandas as pd

from scripts.agents.primary_agent import PrimaryAgent
from scripts.agents.rag_agent import RAGAgent
from scripts.agents.sql_agent import SQLAgent, SQLQueryTool, DBSchemaTool
from scripts.evaluation import evaluate_response
from scripts.report_generation import generate_report
from scripts.config import (
    SQLITE_DB_PATH, CHROMA_DB_PATH, PROJECT_ROOT, PDF_DIR, CSV_DIR,
    USE_OLLAMA, OLLAMA_MODEL, HUGGINGFACE_MODEL, EVALUATION_METRICS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChatbotApp:
    """Main chatbot application class that manages the Gradio interface and agents."""
    
    def __init__(self):
        self.primary_agent = None
        self.rag_agent = None
        self.sql_agent = None
        self.chat_history = []
        self.evaluation_results = {}
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize all agents required for the application."""
        try:
            logger.info("Initializing agents...")
            
            # Initialize the LLM
            from scripts.agents.primary_agent import initialize_llm
            llm = initialize_llm()
            
            # Initialize RAG agent
            self.rag_agent = RAGAgent()
            
            # Initialize SQL agent
            self.sql_agent = SQLAgent(llm)
            
            # Initialize primary agent
            self.primary_agent = PrimaryAgent()
            
            # Register specialized agents with the primary agent
            self.primary_agent.register_specialized_agent(
                name="rag_agent",
                agent=self.rag_agent
            )
            
            self.primary_agent.register_specialized_agent(
                name="sql_agent",
                agent=self.sql_agent
            )
            
            # Add direct tools for testing
            sql_query_tool = SQLQueryTool()
            db_schema_tool = DBSchemaTool()
            
            self.primary_agent.add_tool(sql_query_tool)
            self.primary_agent.add_tool(db_schema_tool)
            
            # Build the primary agent
            self.primary_agent.build_agent()
            
            logger.info("All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def process_query(self, query: str, history=None) -> Dict[str, Any]:
        """
        Process a user query through the primary agent.
        
        Args:
            query: User query string
            history: Chat history for context
            
        Returns:
            Dictionary with agent response and metadata
        """
        if not query.strip():
            return {"response": "Please enter a valid query.", "success": False}
        
        logger.info(f"Processing query: '{query}'")
        
        try:
            # Run the primary agent
            result = self.primary_agent.run(query)
            
            # Add to chat history
            if history is None:
                history = []
            
            history.append((query, result["response"]))
            self.chat_history = history
            
            # Evaluate response (if needed)
            if query.lower().startswith("evaluate:"):
                evaluation_query = query[len("evaluate:"):].strip()
                if evaluation_query:
                    evaluation = evaluate_response(
                        query=evaluation_query,
                        response=result["response"],
                        metrics=EVALUATION_METRICS
                    )
                    result["evaluation"] = evaluation
                    self.evaluation_results[evaluation_query] = evaluation
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def get_db_info(self) -> Dict[str, Any]:
        """Get information about the database schema."""
        try:
            # Get SQL DB info
            if self.sql_agent:
                sql_info = self.sql_agent.get_schema_info()
            else:
                sql_info = {"error": "SQL agent not initialized"}
            
            # Get vector DB info
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
                embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                
                db = Chroma(
                    persist_directory=str(CHROMA_DB_PATH),
                    embedding_function=embedding_function
                )
                
                collections = db._client.list_collections()
                collection_info = {}
                
                for collection in collections:
                    count = db._client.get_collection(collection.name).count()
                    collection_info[collection.name] = {
                        "count": count
                    }
                
                vector_info = {
                    "collections": collection_info
                }
                
            except Exception as e:
                vector_info = {"error": f"Error getting vector DB info: {str(e)}"}
            
            return {
                "sql_database": sql_info,
                "vector_database": vector_info
            }
            
        except Exception as e:
            logger.error(f"Error getting DB info: {str(e)}")
            return {"error": str(e)}
    
    def generate_evaluation_report(self) -> str:
        """Generate a report based on evaluation results."""
        try:
            if not self.evaluation_results:
                return "No evaluation results available."
                
            report = generate_report(self.evaluation_results)
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def build_gradio_interface(self):
        """Build and launch the Gradio interface."""
        with gr.Blocks(title="Q&A Chatbot with RAG and SQL") as interface:
            gr.Markdown("# Q&A Chatbot with RAG and SQL")
            gr.Markdown("Ask questions about the structured data (SQL) or unstructured data (documents)")
            
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(label="Conversation")
                msg = gr.Textbox(label="Your Question", placeholder="Ask something about the data...")
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                with gr.Accordion("Advanced Options", open=False):
                    agent_info = gr.JSON(label="Agent Details", visible=False)
                    eval_checkbox = gr.Checkbox(label="Evaluate Response", value=False)
            
            with gr.Tab("Database Info"):
                db_info_btn = gr.Button("Get Database Info")
                db_info_json = gr.JSON(label="Database Information")
            
            with gr.Tab("Evaluation"):
                eval_report_btn = gr.Button("Generate Evaluation Report")
                eval_report = gr.Markdown(label="Evaluation Report")
            
            # Define callbacks
            def user_input(user_message, history):
                return "", history + [[user_message, None]]
            
            def bot_response(history):
                if history and history[-1][1] is None:
                    query = history[-1][0]
                    result = self.process_query(query, history[:-1])
                    
                    # Update history with the response
                    history[-1][1] = result["response"]
                    
                    # Show additional info if available
                    agent_details = {
                        "agent_used": result.get("agent", "primary_agent"),
                        "success": result.get("success", True),
                    }
                    
                    if "evaluation" in result:
                        agent_details["evaluation"] = result["evaluation"]
                    
                    return history, agent_details
                return history, {}
            
            def get_db_information():
                return self.get_db_info()
            
            def generate_eval_report():
                return self.generate_evaluation_report()
            
            def clear_chat():
                return None, None
            
            # Connect UI components
            submit_btn.click(
                user_input, 
                [msg, chatbot], 
                [msg, chatbot], 
                queue=False
            ).then(
                bot_response,
                [chatbot],
                [chatbot, agent_info]
            )
            
            msg.submit(
                user_input, 
                [msg, chatbot], 
                [msg, chatbot], 
                queue=False
            ).then(
                bot_response,
                [chatbot],
                [chatbot, agent_info]
            )
            
            clear_btn.click(clear_chat, None, [chatbot, agent_info])
            db_info_btn.click(get_db_information, None, db_info_json)
            eval_report_btn.click(generate_eval_report, None, eval_report)
            
            # Add example questions
            gr.Examples(
                examples=[
                    "What tables are in the database?",
                    "What documents do we have about project management?",
                    "What were the sales figures for January 2023?",
                    "Summarize the key points about data governance from our documents.",
                    "Compare the performance of Product A and Product B based on the data.",
                    "What are the main topics covered in our technical documentation?",
                    "Evaluate: What is our revenue trend for the past 6 months?"
                ],
                inputs=msg
            )
            
        return interface
    
    def launch(self, share=False, debug=False):
        """Launch the Gradio application."""
        interface = self.build_gradio_interface()
        interface.launch(share=share, debug=debug)

if __name__ == "__main__":
    logger.info("Starting Q&A Chatbot application")
    
    try:
        # Check if required directories and files exist
        if not os.path.exists(SQLITE_DB_PATH.parent):
            os.makedirs(SQLITE_DB_PATH.parent, exist_ok=True)
            logger.info(f"Created database directory: {SQLITE_DB_PATH.parent}")
        
        if not os.path.exists(CHROMA_DB_PATH):
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            logger.info(f"Created Chroma DB directory: {CHROMA_DB_PATH}")
        
        # Initialize and launch app
        app = ChatbotApp()
        app.launch(debug=True)
        
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        raise
