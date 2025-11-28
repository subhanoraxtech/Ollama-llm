"""
CSV file handling utilities with encoding detection and error handling.
"""
import pandas as pd
import chardet
from typing import Optional, Dict, Any
import io


def detect_encoding(file_bytes: bytes) -> str:
    """
    Detect the encoding of a file.
    
    Args:
        file_bytes: File content as bytes
        
    Returns:
        Detected encoding string
    """
    result = chardet.detect(file_bytes)
    return result['encoding'] or 'utf-8'


def read_csv_safe(file_path_or_buffer, **kwargs) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely read a CSV file with automatic encoding detection and error handling.
    
    Args:
        file_path_or_buffer: File path or buffer object
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        Tuple of (DataFrame, error_message). If successful, error_message is None.
    """
    try:
        # If it's a file-like object, read bytes for encoding detection
        if hasattr(file_path_or_buffer, 'read'):
            # Read the content
            content = file_path_or_buffer.read()
            
            # Detect encoding if bytes
            if isinstance(content, bytes):
                encoding = detect_encoding(content)
                file_buffer = io.BytesIO(content)
            else:
                encoding = 'utf-8'
                file_buffer = io.StringIO(content)
            
            # Try reading with detected encoding
            try:
                df = pd.read_csv(file_buffer, encoding=encoding, **kwargs)
                return df, None
            except Exception as e:
                # Try with utf-8 as fallback
                if isinstance(content, bytes):
                    file_buffer = io.BytesIO(content)
                else:
                    file_buffer = io.StringIO(content)
                df = pd.read_csv(file_buffer, encoding='utf-8', encoding_errors='ignore', **kwargs)
                return df, None
        
        # If it's a file path
        else:
            # Read file bytes for encoding detection
            with open(file_path_or_buffer, 'rb') as f:
                raw_data = f.read()
            
            encoding = detect_encoding(raw_data)
            
            # Try reading with detected encoding
            try:
                df = pd.read_csv(file_path_or_buffer, encoding=encoding, **kwargs)
                return df, None
            except Exception as e:
                # Try with utf-8 as fallback
                df = pd.read_csv(file_path_or_buffer, encoding='utf-8', encoding_errors='ignore', **kwargs)
                return df, None
                
    except Exception as e:
        return None, f"Error reading CSV: {str(e)}"


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, Optional[str]]:
    """
    Validate that a DataFrame is suitable for processing.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if not isinstance(df, pd.DataFrame):
        return False, "Object is not a DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    
    return True, None


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with DataFrame information
    """
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'null_counts': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
    }
    
    # Add sample data
    info['sample_data'] = df.head(5).to_dict('records')
    
    # Add statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        info['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    return info


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer the semantic type of each column.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary mapping column names to inferred types
    """
    from .validators import detect_column_type
    
    column_types = {}
    for col in df.columns:
        column_types[col] = detect_column_type(df[col])
    
    return column_types


def prepare_dataframe_for_display(df: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
    """
    Prepare a DataFrame for display in UI (limit rows, format values).
    
    Args:
        df: DataFrame to prepare
        max_rows: Maximum number of rows to include
        
    Returns:
        Prepared DataFrame
    """
    # Limit rows
    display_df = df.head(max_rows).copy()
    
    # Format numeric columns to 2 decimal places
    numeric_cols = display_df.select_dtypes(include=['float']).columns
    for col in numeric_cols:
        display_df[col] = display_df[col].round(2)
    
    return display_df


def export_dataframe(df: pd.DataFrame, file_path: str, format: str = 'csv', **kwargs) -> tuple[bool, Optional[str]]:
    """
    Export DataFrame to various formats.
    
    Args:
        df: DataFrame to export
        file_path: Output file path
        format: Export format ('csv', 'excel', 'json')
        **kwargs: Additional arguments for export function
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False, **kwargs)
        elif format.lower() in ['excel', 'xlsx']:
            df.to_excel(file_path, index=False, **kwargs)
        elif format.lower() == 'json':
            df.to_json(file_path, orient='records', **kwargs)
        else:
            return False, f"Unsupported format: {format}"
        
        return True, None
    except Exception as e:
        return False, f"Error exporting to {format}: {str(e)}"
