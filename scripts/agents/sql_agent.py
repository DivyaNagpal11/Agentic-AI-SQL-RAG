"""
SQL Agent responsible for querying structured data stored in SQLite.
"""
import logging
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase

# Use correct import path for BaseAgent
from scripts.agents.primary_agent import BaseAgent, initialize_llm
from scripts.config import SQLITE_DB_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLAgent(BaseAgent):
    """
    Agent specialized for handling SQL queries against structured data.
    """
    
    def __init__(self):
        """
        Initialize the SQL agent.
        """
        super().__init__(
            name="sql_agent",
            description="Agent that can query structured data stored in SQLite database."
        )
        self.db = SQLDatabase.from_uri(f"sqlite:///{SQLITE_DB_PATH}")
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self._setup_tools()
        
    def _setup_tools(self) -> None:
        """Set up the tools for this agent."""
        # Add all toolkit tools to this agent
        for tool in self.toolkit.get_tools():
            self.add_tool(tool)
        
        # Add custom schema tool
        schema_tool = DBSchemaTool()
        self.add_tool(schema_tool)
        
    def build_agent(self) -> None:
        """Build the SQL agent with the database toolkit."""
        if not self.tools:
            raise ValueError(f"No tools registered for {self.name} agent. Add tools before building.")
        
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the SQL agent to query the database.
        
        Args:
            query: The natural language query to process
            
        Returns:
            Dictionary with the agent's response and metadata
        """
        if not self.agent_executor:
            self.build_agent()
            
        logger.info(f"Running SQL agent with query: '{query}'")
        
        try:
            response = self.agent_executor.run(
                f"Answer the following question using SQL queries on the database: {query}"
            )
            
            return {
                "response": response,
                "agent": self.name,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in SQL agent: {str(e)}")
            
            return {
                "response": f"I encountered an error while querying the database: {str(e)}",
                "agent": self.name,
                "success": False,
                "error": str(e)
            }
            
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the database schema."""
        try:
            tables = self.db.get_usable_table_names()
            schema = {}
            
            for table in tables:
                columns = self.db.get_table_info(table)
                schema[table] = columns
            
            return {
                "tables": tables,
                "schema": schema
            }
            
        except Exception as e:
            logger.error(f"Error getting schema info: {str(e)}")
            return {"error": str(e)}

class SQLQueryTool(BaseTool):
    """Tool for running SQL queries on a database."""
    
    name = "sql_query_tool"
    description = "Useful for running SQL queries against a database to retrieve or analyze structured data."
    
    class SQLQueryToolArgs(BaseModel):
        query: str = Field(..., description="The SQL query to execute")
    
    args_schema = SQLQueryToolArgs
    
    def __init__(self, db_path: str = None):
        """Initialize the SQL query tool."""
        super().__init__()
        if db_path is None:
            db_path = str(SQLITE_DB_PATH)
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    def _run(self, query: str) -> str:
        """Run the SQL query and return the results."""
        try:
            result = self.db.run(query)
            return result
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"

class DBSchemaTool(BaseTool):
    """Tool for getting database schema information."""
    
    name = "db_schema_tool"
    description = "Useful for retrieving database schema information including tables and their columns."
    
    def __init__(self, db_path: str = None):
        """Initialize the schema tool."""
        super().__init__()
        if db_path is None:
            db_path = str(SQLITE_DB_PATH)
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    
    def _run(self, _: str = None) -> str:
        """Get and return the database schema."""
        try:
            tables = self.db.get_usable_table_names()
            schema_info = []
            
            for table in tables:
                columns = self.db.get_table_info(table)
                schema_info.append(f"Table: {table}\nColumns: {columns}\n")
            
            return "\n".join(schema_info)
        except Exception as e:
            return f"Error retrieving database schema: {str(e)}"

if __name__ == "__main__":
    # Test the SQLQueryTool and SQLAgent
    
    # Create and test SQL agent
    sql_agent = SQLAgent()
    
    # Test query
    test_query = "What tables are in the database and what columns do they have?"
    result = sql_agent.run(test_query)
    
    print(f"Test result: {result}")