"""
Base agent class for all data analysis agents.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import logging
from datetime import datetime


class BaseAgent(ABC):
    """
    Base class for all data analysis agents.
    Provides common functionality for logging, error handling, and result formatting.
    """
    
    def __init__(self, name: str):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
        """
        self.name = name
        self.logger = self._setup_logger()
        self.operation_history = []
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger for the agent.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        # Create console handler if not already exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def log_operation(self, operation: str, details: Dict[str, Any]):
        """
        Log an operation to the history.
        
        Args:
            operation: Name of the operation
            details: Details about the operation
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        }
        self.operation_history.append(entry)
        self.logger.info(f"Operation: {operation} - {details}")
    
    def create_result(
        self,
        success: bool,
        data: Optional[Any] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized result dictionary.
        
        Args:
            success: Whether the operation was successful
            data: Result data (DataFrame, dict, etc.)
            message: Human-readable message
            metadata: Additional metadata about the operation
            
        Returns:
            Standardized result dictionary
        """
        result = {
            'success': success,
            'message': message,
            'data': data,
            'metadata': metadata or {},
            'agent': self.name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add DataFrame info if data is a DataFrame
        if isinstance(data, pd.DataFrame):
            result['metadata']['rows'] = len(data)
            result['metadata']['columns'] = len(data.columns)
            result['metadata']['column_names'] = data.columns.tolist()
        
        return result
    
    def handle_error(self, operation: str, error: Exception) -> Dict[str, Any]:
        """
        Handle an error and create an error result.
        
        Args:
            operation: Name of the operation that failed
            error: The exception that occurred
            
        Returns:
            Error result dictionary
        """
        error_msg = f"Error in {operation}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        return self.create_result(
            success=False,
            message=error_msg,
            metadata={'operation': operation, 'error_type': type(error).__name__}
        )
    
    def validate_dataframe(self, df: Any, param_name: str = "dataframe") -> tuple[bool, Optional[str]]:
        """
        Validate that a parameter is a valid DataFrame.
        
        Args:
            df: Object to validate
            param_name: Name of the parameter (for error messages)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None:
            return False, f"{param_name} is None"
        
        if not isinstance(df, pd.DataFrame):
            return False, f"{param_name} must be a pandas DataFrame"
        
        if df.empty:
            return False, f"{param_name} is empty"
        
        return True, None
    
    def validate_column_exists(self, df: pd.DataFrame, column: str) -> tuple[bool, Optional[str]]:
        """
        Validate that a column exists in the DataFrame.
        
        Args:
            df: DataFrame to check
            column: Column name to validate
            
        Returns:
            Tuple of (exists, error_message)
        """
        if column not in df.columns:
            return False, f"Column '{column}' not found. Available columns: {', '.join(df.columns)}"
        return True, None
    
    @abstractmethod
    def get_available_operations(self) -> Dict[str, str]:
        """
        Get a dictionary of available operations and their descriptions.
        
        Returns:
            Dictionary mapping operation names to descriptions
        """
        pass
    
    def get_operation_history(self) -> list:
        """
        Get the operation history for this agent.
        
        Returns:
            List of operation history entries
        """
        return self.operation_history
    
    def clear_history(self):
        """Clear the operation history."""
        self.operation_history = []
        self.logger.info("Operation history cleared")
