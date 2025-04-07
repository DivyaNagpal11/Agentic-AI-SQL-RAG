"""
Evaluation script using RAGAS to assess the performance of agents and retrieval strategies.
"""
import logging
import pandas as pd
from typing import List, Dict, Any, Union, Optional
import json
from pathlib import Path
import datetime

from langchain_core.language_models import BaseLanguageModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.metrics.critique import harmfulness
from ragas import evaluate

try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )

except ImportError:
    logging.error("RAGAS not found. Install with 'pip install ragas'")
    raise

from scripts.config import (
    CHROMA_DB_PATH,
    EVALUATION_METRICS,
    USE_OLLAMA,
    OLLAMA_MODEL,
    PROJECT_ROOT
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_embedding_function():
    """Initialize and return the embedding function based on configuration."""
    if USE_OLLAMA:
        try:
            return OllamaEmbeddings(model=OLLAMA_MODEL)
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embeddings: {e}")
            logger.info("Falling back to HuggingFace embeddings")
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

class RagasEvaluator:
    """
    Evaluator class using RAGAS to assess RAG and Agent performance.
    """
    
    def __init__(self, llm: BaseLanguageModel):
        """
        Initialize the evaluator with the given language model.
        
        Args:
            llm: Language model to use for evaluation
        """
        self.llm = llm
        self.embedding_function = initialize_embedding_function()
        self.results_dir = PROJECT_ROOT / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Define metrics to use
        self.metrics = []
        for metric_name in EVALUATION_METRICS:
            if metric_name == "faithfulness":
                self.metrics.append(faithfulness)
            elif metric_name == "answer_relevancy":
                self.metrics.append(answer_relevancy)
            elif metric_name == "context_precision":
                self.metrics.append(context_precision)
            elif metric_name == "context_recall":
                self.metrics.append(context_recall)
            elif metric_name == "harmfulness":
                self.metrics.append(harmfulness)
    
    def prepare_evaluation_data(
        self, 
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]] = None,
        ground_truths: List[str] = None
    ) -> Dict[str, List]:
        """
        Prepare evaluation data format for RAGAS.
        
        Args:
            questions: List of question strings
            answers: List of answer strings
            contexts: List of lists of context strings
            ground_truths: List of ground truth answer strings
            
        Returns:
            Dictionary with formatted evaluation data
        """
        data = {
            "question": questions,
            "answer": answers,
        }
        
        if contexts:
            data["contexts"] = contexts
            
        if ground_truths:
            data["ground_truths"] = ground_truths
            
        return data
    
    def evaluate_rag(
        self, 
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate RAG system performance using RAGAS.
        
        Args:
            questions: List of question strings
            answers: List of answer strings
            contexts: List of lists of context strings used to generate answers
            ground_truths: Optional list of ground truth answers
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Evaluating RAG performance on {len(questions)} questions")
        
        try:
            # Prepare evaluation data
            eval_data = self.prepare_evaluation_data(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths
            )
            
            # Run evaluation with RAGAS
            result = evaluate(
                eval_data,
                metrics=self.metrics
            )
            
            # Save results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = self.results_dir / f"rag_evaluation_{timestamp}.csv"
            result.to_csv(result_path)
            
            logger.info(f"RAG evaluation complete. Results saved to {result_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating RAG: {str(e)}")
            return pd.DataFrame({"error": [str(e)]})
    
    def evaluate_agent(
        self, 
        agent,
        test_questions: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate agent performance using RAGAS.
        
        Args:
            agent: The agent to evaluate
            test_questions: List of test question strings
            ground_truths: Optional list of ground truth answers
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Evaluating agent performance on {len(test_questions)} questions")
        
        try:
            # Get agent responses
            answers = []
            for question in test_questions:
                result = agent.run(question)
                answers.append(result["response"])
            
            # For RAGAS evaluation, we need contexts too
            # This is a simplification; in practice, you'd extract contexts from the agent's reasoning
            contexts = [["Context not available for agent evaluation"]] * len(test_questions)
            
            # Prepare evaluation data
            eval_data = self.prepare_evaluation_data(
                questions=test_questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths
            )
            
            # Run evaluation with subset of metrics that don't require real contexts
            metrics = [faithfulness, answer_relevancy]
            
            result = evaluate(
                eval_data,
                metrics=metrics
            )
            
            # Save results
            agent_name = getattr(agent, "name", "agent")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = self.results_dir / f"{agent_name}_evaluation_{timestamp}.csv"
            result.to_csv(result_path)
            
            logger.info(f"Agent evaluation complete. Results saved to {result_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating agent: {str(e)}")
            return pd.DataFrame({"error": [str(e)]})
    
    def evaluate_retrieval(
        self,
        questions: List[str],
        collection_name: str = "documents",
        top_k: int = 5,
        ground_truth_docs: Optional[List[List[str]]] = None
    ) -> pd.DataFrame:
        """
        Evaluate retrieval performance using RAGAS.
        
        Args:
            questions: List of question strings
            collection_name: Name of the ChromaDB collection
            top_k: Number of documents to retrieve
            ground_truth_docs: Optional list of lists of ground truth document IDs
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Evaluating retrieval performance on {len(questions)} questions")
        
        try:
            # Initialize Chroma
            db = Chroma(
                persist_directory=str(CHROMA_DB_PATH),
                embedding_function=self.embedding_function,
                collection_name=collection_name
            )
            
            # Retrieve documents for each question
            retrieved_contexts = []
            for question in questions:
                docs = db.similarity_search(question, k=top_k)
                retrieved_contexts.append([doc.page_content for doc in docs])
            
            # For retrieval evaluation, we don't need answers
            # but RAGAS requires them, so we'll use placeholders
            placeholder_answers = ["Placeholder answer"] * len(questions)
            
            # Prepare evaluation data (without ground truth for now)
            eval_data = self.prepare_evaluation_data(
                questions=questions,
                answers=placeholder_answers,
                contexts=retrieved_contexts
            )
            
            # Use only context-related metrics
            metrics = [context_precision, context_recall]
            
            # If ground truth docs are provided, we can evaluate more accurately
            if ground_truth_docs:
                eval_data["ground_truths"] = ground_truth_docs
            
            # Run evaluation with RAGAS
            result = evaluate(
                eval_data,
                metrics=metrics
            )
            
            # Save results
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = self.results_dir / f"retrieval_evaluation_{timestamp}.csv"
            result.to_csv(result_path)
            
            logger.info(f"Retrieval evaluation complete. Results saved to {result_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval: {str(e)}")
            return pd.DataFrame({"error": [str(e)]})

if __name__ == "__main__":
    # Test the evaluator
    
    # Use a simple LLM for testing
    llm = Ollama(model="llama3")
    
    # Create evaluator
    evaluator = RagasEvaluator(llm)
    
    # Test data
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?"
    ]
    
    answers = [
        "The capital of France is Paris.",
        "Romeo and Juliet was written by William Shakespeare."
    ]
    
    contexts = [
        ["Paris is the capital of France.", "France is located in Western Europe."],
        ["William Shakespeare wrote Romeo and Juliet in the late 16th century.", "Romeo and Juliet is a tragedy."]
    ]
    
    # Test RAG evaluation
    rag_results = evaluator.evaluate_rag(questions, answers, contexts)
    print(f"RAG evaluation results: {rag_results}")