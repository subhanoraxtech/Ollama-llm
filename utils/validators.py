"""
Data validation utilities for the data analysis agent system.
"""
import re
from typing import Union
import pandas as pd


def validate_email(email: str) -> bool:
    """
    Validate email format using regex.
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid email format, False otherwise
    """
    if not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_phone(phone: Union[str, int]) -> bool:
    """
    Validate phone number format.
    Accepts various formats: (123) 456-7890, 123-456-7890, 1234567890, +1234567890
    
    Args:
        phone: Phone number to validate
        
    Returns:
        True if valid phone format, False otherwise
    """
    if pd.isna(phone):
        return False
    
    phone_str = str(phone).strip()
    
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)\+\.]', '', phone_str)
    
    # Check if it's all digits and has reasonable length (7-15 digits)
    return cleaned.isdigit() and 7 <= len(cleaned) <= 15


def validate_date(date_str: str, formats: list = None) -> bool:
    """
    Validate date format.
    
    Args:
        date_str: Date string to validate
        formats: List of date formats to try (default: common formats)
        
    Returns:
        True if valid date format, False otherwise
    """
    if not isinstance(date_str, str):
        return False
    
    if formats is None:
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%d-%m-%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y'
        ]
    
    for fmt in formats:
        try:
            pd.to_datetime(date_str, format=fmt)
            return True
        except:
            continue
    
    return False


def validate_numeric(value: Union[str, int, float]) -> bool:
    """
    Validate if value can be converted to numeric.
    
    Args:
        value: Value to validate
        
    Returns:
        True if can be converted to numeric, False otherwise
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_valid_column_name(name: str) -> bool:
    """
    Check if column name is valid (no special characters except underscore).
    
    Args:
        name: Column name to validate
        
    Returns:
        True if valid column name, False otherwise
    """
    if not isinstance(name, str) or not name:
        return False
    
    # Allow letters, numbers, and underscores
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))


def standardize_column_name(name: str) -> str:
    """
    Standardize column name to lowercase with underscores.
    
    Args:
        name: Column name to standardize
        
    Returns:
        Standardized column name
    """
    # Convert to string and strip whitespace
    name = str(name).strip()
    
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s\-]+', '_', name)
    
    # Remove special characters except underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    
    # Convert to lowercase
    name = name.lower()
    
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = 'col_' + name
    
    return name if name else 'unnamed'


def detect_column_type(series: pd.Series) -> str:
    """
    Detect the most appropriate data type for a pandas Series.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        Detected type: 'numeric', 'datetime', 'email', 'phone', 'categorical', 'text'
    """
    # Remove null values for analysis
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return 'unknown'
    
    # Check if numeric
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    
    # Convert to string for pattern matching
    sample = non_null.head(100).astype(str)
    
    # Check for email pattern
    email_count = sum(sample.apply(validate_email))
    if email_count / len(sample) > 0.8:
        return 'email'
    
    # Check for phone pattern
    phone_count = sum(sample.apply(validate_phone))
    if phone_count / len(sample) > 0.8:
        return 'phone'
    
    # Check for datetime
    try:
        pd.to_datetime(non_null.head(100))
        return 'datetime'
    except:
        pass
    
    # Check if categorical (low cardinality)
    unique_ratio = len(non_null.unique()) / len(non_null)
    if unique_ratio < 0.05 or len(non_null.unique()) < 20:
        return 'categorical'
    
    return 'text'
