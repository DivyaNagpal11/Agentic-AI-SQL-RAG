"""
Primary agent that routes queries to the appropriate specialized agent.
"""
import logging
from typing import Dict, Any, List, Optional, Callable
from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.manager import CallbackManager
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.base import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain.tools import Tool
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate
from scripts.config import USE_OLLAMA, OLLAMA_MODEL, HUGGINGFACE_MODEL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_llm() -> BaseLanguageModel:
    """Initialize and return the appropriate LLM based on configuration."""
    if USE_OLLAMA:
        try:
            return Ollama(model=OLLAMA_MODEL)
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            logger.info("Falling back to HuggingFace model")
            return HuggingFaceEndpoint(repo_id=HUGGINGFACE_MODEL)
    else:
        try:
            return HuggingFaceEndpoint(repo_id=HUGGINGFACE_MODEL)
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace LLM: {e}")
            # Try one more fallback
            try:
                return HuggingFaceHub(repo_id="google/flan-t5-base")
            except:
                logger.error("All LLM initialization attempts failed")
                raise RuntimeError("Unable to initialize any LLM")

class BaseAgent:
    """Base class for specialized agents."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.llm = initialize_llm()
        self.tools = []
        self.agent_executor = None
        
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)
        
    def build_agent(self) -> None:
        """Build the agent executor with the agent's tools."""
        if not self.tools:
            raise ValueError(f"No tools registered for {self.name} agent. Add tools before building.")
        
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
    def run(self, query: str) -> Dict[str, Any]:
        """Run the specialized agent on the given query."""
        if not self.agent_executor:
            self.build_agent()
            
        logger.info(f"Running {self.name} agent with query: '{query}'")
        
        try:
            response = self.agent_executor.run(query)
            
            return {
                "response": response,
                "agent": self.name,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in {self.name} agent: {str(e)}")
            
            return {
                "response": f"Error in {self.name} agent: {str(e)}",
                "agent": self.name,
                "success": False,
                "error": str(e)
            }
    
    def get_description(self) -> str:
        """Get the description of this specialized agent."""
        return self.description
    
    def should_return_direct(self) -> bool:
        """Whether the response from this agent should be returned directly."""
        return True

class PrimaryAgent:
    """
    Primary agent that routes queries to specialized agents based on the query type.
    """
    
    def __init__(self):
        self.llm = initialize_llm()
        self.tools = []
        self.agent_executor = None
        self.specialized_agents = {}
        
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)
        
    def register_specialized_agent(self, name: str, agent: 'BaseAgent') -> None:
        """Register a specialized agent that can be called by the primary agent."""
        self.specialized_agents[name] = agent
        
        # Create a tool to invoke this specialized agent
        
        tool = Tool(
            name=name,
            func=agent.run,
            description=agent.get_description(),
            return_direct=agent.should_return_direct()
        )
        
        self.add_tool(tool)
        
    def build_agent(self) -> None:
        """Build the agent executor with the registered tools."""
        if not self.tools:
            raise ValueError("No tools registered. Add tools before building the agent.")
        
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the primary agent on the given query.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with agent response and metadata
        """
        if not self.agent_executor:
            self.build_agent()
            
        logger.info(f"Running primary agent with query: '{query}'")
        
        try:
            response = self.agent_executor.run(query)
            
            return {
                "response": response,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in primary agent: {str(e)}")
            
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def get_tool_descriptions(self) -> List[Dict[str, str]]:
        """Get descriptions of all registered tools."""
        return [{"name": tool.name, "description": tool.description} for tool in self.tools]

# Create a tool selector function using LangGraph
def create_tool_selector(tools: List[BaseTool], llm: BaseLanguageModel) -> Callable:
    """
    Create a function that selects the appropriate tool based on the query.
    
    Args:
        tools: List of available tools
        llm: Language model to use for selection
        
    Returns:
        Function that selects a tool based on query
    """
    
    # Define the state
    class AgentState(dict):
        query: str
        tool_name: Optional[str] = None
    
    # Create prompt for tool selection
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    tool_names = [tool.name for tool in tools]
    
    tool_selector_prompt = ChatPromptTemplate.from_template("""
    You are a tool selection AI. Based on the user query, select the most appropriate tool from the available tools.
    
    Available tools:
    {tool_descriptions}
    
    User query: {query}
    
    Select the tool that best matches this query. Respond with just the name of the tool.
    If none of the tools are appropriate, respond with "none".
    """)
    
    # Define the selection node
    def select_tool(state: AgentState) -> AgentState:
        state_dict = state.copy()
        
        response = llm.invoke(
            tool_selector_prompt.format(
                tool_descriptions=tool_descriptions,
                query=state["query"]
            )
        )
        
        selected_tool = response.content.strip().lower()
        
        # Check if selected tool is in available tools
        if selected_tool in tool_names:
            state_dict["tool_name"] = selected_tool
        else:
            state_dict["tool_name"] = None
            
        return state_dict
    
    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("select_tool", select_tool)
    workflow.add_edge("select_tool", END)
    
    # Compile the graph
    app = workflow.compile()
    
    # Return a function that runs the graph
    def selector(query: str) -> Optional[str]:
        result = app.invoke({"query": query})
        return result["tool_name"]
    
    return selector

def build_primary_agent_with_langgraph() -> Callable:
    """
    Build a more advanced primary agent using LangGraph for reasoning.
    
    Returns:
        Function that runs the primary agent
    """
    
    # First initialize all the tools and agents
    llm = initialize_llm()
    
    # Define state
    class AgentState(dict):
        messages: List[Dict]
        query: str
        tool_calls: Optional[List] = None
        current_tool: Optional[str] = None
        
    # Node for analyzing the query
    def analyze_query(state: AgentState) -> AgentState:
        state_dict = state.copy()
        # Create prompt for analyzing the query
        analyze_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant that helps route user queries to the appropriate specialized agent.
        
        User query: {query}
        
        Is this query about:
        1. Retrieving information from unstructured documents (use RAG agent)
        2. Querying structured data in a database (use SQL agent)
        3. General question (handle directly)
        
        Respond with just the type of query (1, 2, or 3).
        """)
        
        response = llm.invoke(analyze_prompt.format(query=state["query"]))
        query_type = response.content.strip()
        
        if "1" in query_type:
            state_dict["current_tool"] = "rag_agent"
        elif "2" in query_type:
            state_dict["current_tool"] = "sql_agent"
        else:
            state_dict["current_tool"] = None
            
        return state_dict
    
    # Node for invoking tools
    def invoke_tool(state: AgentState) -> AgentState:
        state_dict = state.copy()
        
        if state_dict["current_tool"]:
            # In a real implementation, you would have registered tools
            # This is a placeholder for tool execution
            tool_message = f"Selected tool: {state_dict['current_tool']}"
            state_dict["messages"].append({"role": "function", "content": tool_message})
        else:
            # No specific tool needed
            state_dict["messages"].append({"role": "system", "content": "No specialized agent required."})
            
        return state_dict
    
    # Node for generating response
    def generate_response(state: AgentState) -> AgentState:
        state_dict = state.copy()
        
        # Construct conversation history
        history = state_dict["messages"]
        query = state_dict["query"]
        
        # Create prompt for final response
        response_prompt = ChatPromptTemplate.from_template("""
        Given the following conversation history and the current query, generate a helpful response.
        
        History:
        {history}
        
        Current query: {query}
        
        Response:
        """)
        
        response = llm.invoke(response_prompt.format(
            history="\n".join([f"{m['role']}: {m['content']}" for m in history]),
            query=query
        ))
        
        state_dict["messages"].append({"role": "assistant", "content": response.content})
        state_dict["response"] = response.content
        
        return state_dict
    
    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("invoke_tool", invoke_tool)
    workflow.add_node("respond", generate_response)
    
    # Add edges
    workflow.add_edge("analyze", "invoke_tool")
    workflow.add_edge("invoke_tool", "respond")
    workflow.add_edge("respond", END)
    
    # Compile graph
    app = workflow.compile()
    
    # Return function
    def run_agent(query: str) -> Dict[str, Any]:
        try:
            result = app.invoke({
                "messages": [],
                "query": query
            })
            return {
                "response": result["response"],
                "success": True
            }
        except Exception as e:
            logger.error(f"Error in LangGraph agent: {str(e)}")
            return {
                "response": f"Error: {str(e)}",
                "success": False
            }
    
    return run_agent

if __name__ == "__main__":
    # Test the primary agent
    primary_agent = PrimaryAgent()
    test_tool = Tool(
        name="echo",
        func=lambda x: f"Echo: {x}",
        description="Repeats back the input"
    )
    primary_agent.add_tool(test_tool)
    primary_agent.build_agent()
    test_query = "Echo this message back to me"
    result = primary_agent.run(test_query)
    print(f"Test result: {result}")
