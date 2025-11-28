"""
Data Cleaning Agent - handles all data cleaning operations.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
import re
from .base_agent import BaseAgent
from utils.validators import validate_email, validate_phone, standardize_column_name


class CleaningAgent(BaseAgent):
    """Agent for data cleaning operations."""
    
    def __init__(self):
        super().__init__("CleaningAgent")
    
    def get_available_operations(self) -> Dict[str, str]:
        """Get available cleaning operations."""
        return {
            'remove_duplicates': 'Remove duplicate rows from the dataset',
            'fix_email_formatting': 'Fix and validate email addresses',
            'fix_phone_formatting': 'Fix and validate phone numbers',
            'fix_name_formatting': 'Standardize name formatting (title case)',
            'trim_whitespace': 'Remove leading/trailing whitespace from all text columns',
            'normalize_case': 'Convert text to uppercase or lowercase',
            'handle_missing_values': 'Fill, drop, or replace missing values',
            'standardize_columns': 'Rename columns to standard format (lowercase_with_underscores)',
            'remove_empty_rows': 'Remove rows that are completely empty',
            'remove_empty_columns': 'Remove columns that are completely empty'
        }
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> Dict[str, Any]:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for identifying duplicates (None = all columns)
            keep: Which duplicates to keep ('first', 'last', False to remove all)
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            initial_rows = len(df)
            cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
            removed_rows = initial_rows - len(cleaned_df)
            
            self.log_operation('remove_duplicates', {
                'initial_rows': initial_rows,
                'removed_rows': removed_rows,
                'final_rows': len(cleaned_df)
            })
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=f"Removed {removed_rows} duplicate rows",
                metadata={'removed_rows': removed_rows, 'initial_rows': initial_rows}
            )
        
        except Exception as e:
            return self.handle_error('remove_duplicates', e)
    
    def fix_email_formatting(
        self,
        df: pd.DataFrame,
        column: str,
        remove_invalid: bool = False
    ) -> Dict[str, Any]:
        """
        Fix and validate email addresses.
        
        Args:
            df: Input DataFrame
            column: Column containing email addresses
            remove_invalid: If True, remove rows with invalid emails
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, column)
            if not exists:
                return self.create_result(False, message=error)
            
            cleaned_df = df.copy()
            
            # Clean emails: strip whitespace, lowercase
            cleaned_df[column] = cleaned_df[column].astype(str).str.strip().str.lower()
            
            # Validate emails
            valid_mask = cleaned_df[column].apply(validate_email)
            invalid_count = (~valid_mask).sum()
            
            if remove_invalid:
                cleaned_df = cleaned_df[valid_mask]
                message = f"Fixed email formatting and removed {invalid_count} invalid emails"
            else:
                message = f"Fixed email formatting. Found {invalid_count} invalid emails"
            
            self.log_operation('fix_email_formatting', {
                'column': column,
                'invalid_count': invalid_count,
                'removed': remove_invalid
            })
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=message,
                metadata={'invalid_count': invalid_count, 'removed_invalid': remove_invalid}
            )
        
        except Exception as e:
            return self.handle_error('fix_email_formatting', e)
    
    def fix_phone_formatting(
        self,
        df: pd.DataFrame,
        column: str,
        format_style: str = 'standard',
        remove_invalid: bool = False
    ) -> Dict[str, Any]:
        """
        Fix and validate phone numbers.
        
        Args:
            df: Input DataFrame
            column: Column containing phone numbers
            format_style: 'standard' (123-456-7890), 'dots' (123.456.7890), 'plain' (1234567890)
            remove_invalid: If True, remove rows with invalid phone numbers
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, column)
            if not exists:
                return self.create_result(False, message=error)
            
            cleaned_df = df.copy()
            
            def format_phone(phone):
                if pd.isna(phone):
                    return phone
                
                # Remove all non-digits
                digits = re.sub(r'\D', '', str(phone))
                
                if not validate_phone(phone):
                    return phone  # Return original if invalid
                
                # Format based on style
                if len(digits) == 10:
                    if format_style == 'standard':
                        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
                    elif format_style == 'dots':
                        return f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"
                    else:  # plain
                        return digits
                elif len(digits) == 11 and digits[0] == '1':
                    # US number with country code
                    if format_style == 'standard':
                        return f"+1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
                    elif format_style == 'dots':
                        return f"+1.{digits[1:4]}.{digits[4:7]}.{digits[7:]}"
                    else:
                        return digits
                
                return phone  # Return original if unusual length
            
            cleaned_df[column] = cleaned_df[column].apply(format_phone)
            
            # Count invalid
            valid_mask = cleaned_df[column].apply(validate_phone)
            invalid_count = (~valid_mask).sum()
            
            if remove_invalid:
                cleaned_df = cleaned_df[valid_mask]
                message = f"Fixed phone formatting and removed {invalid_count} invalid numbers"
            else:
                message = f"Fixed phone formatting. Found {invalid_count} invalid numbers"
            
            self.log_operation('fix_phone_formatting', {
                'column': column,
                'format_style': format_style,
                'invalid_count': invalid_count
            })
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=message,
                metadata={'invalid_count': invalid_count, 'format_style': format_style}
            )
        
        except Exception as e:
            return self.handle_error('fix_phone_formatting', e)
    
    def fix_name_formatting(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Standardize name formatting (title case, trim whitespace).
        
        Args:
            df: Input DataFrame
            columns: Column(s) containing names
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            if isinstance(columns, str):
                columns = [columns]
            
            cleaned_df = df.copy()
            
            for column in columns:
                exists, error = self.validate_column_exists(cleaned_df, column)
                if not exists:
                    return self.create_result(False, message=error)
                
                # Title case and strip whitespace
                cleaned_df[column] = cleaned_df[column].astype(str).str.strip().str.title()
            
            self.log_operation('fix_name_formatting', {'columns': columns})
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=f"Fixed name formatting for {len(columns)} column(s)",
                metadata={'columns': columns}
            )
        
        except Exception as e:
            return self.handle_error('fix_name_formatting', e)
    
    def trim_whitespace(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Remove leading/trailing whitespace from all text columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            cleaned_df = df.copy()
            
            # Get text columns
            text_columns = cleaned_df.select_dtypes(include=['object']).columns
            
            # Strip whitespace
            for col in text_columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            
            self.log_operation('trim_whitespace', {'columns_processed': len(text_columns)})
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=f"Trimmed whitespace from {len(text_columns)} text columns",
                metadata={'columns_processed': list(text_columns)}
            )
        
        except Exception as e:
            return self.handle_error('trim_whitespace', e)
    
    def normalize_case(
        self,
        df: pd.DataFrame,
        columns: Union[str, List[str], None] = None,
        case: str = 'lower'
    ) -> Dict[str, Any]:
        """
        Convert text to uppercase or lowercase.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to normalize (None = all text columns)
            case: 'lower' or 'upper'
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            cleaned_df = df.copy()
            
            if columns is None:
                columns = cleaned_df.select_dtypes(include=['object']).columns.tolist()
            elif isinstance(columns, str):
                columns = [columns]
            
            for column in columns:
                exists, error = self.validate_column_exists(cleaned_df, column)
                if not exists:
                    return self.create_result(False, message=error)
                
                if case.lower() == 'lower':
                    cleaned_df[column] = cleaned_df[column].astype(str).str.lower()
                elif case.lower() == 'upper':
                    cleaned_df[column] = cleaned_df[column].astype(str).str.upper()
                else:
                    return self.create_result(False, message=f"Invalid case option: {case}")
            
            self.log_operation('normalize_case', {'columns': columns, 'case': case})
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=f"Normalized case to {case} for {len(columns)} column(s)",
                metadata={'columns': columns, 'case': case}
            )
        
        except Exception as e:
            return self.handle_error('normalize_case', e)
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'drop',
        fill_value: Any = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Handle missing values.
        
        Args:
            df: Input DataFrame
            strategy: 'drop', 'fill', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
            fill_value: Value to use when strategy is 'fill'
            columns: Specific columns to process (None = all columns)
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            cleaned_df = df.copy()
            initial_rows = len(cleaned_df)
            missing_before = cleaned_df.isnull().sum().sum()
            
            if columns:
                target_df = cleaned_df[columns]
            else:
                target_df = cleaned_df
            
            if strategy == 'drop':
                cleaned_df = cleaned_df.dropna(subset=columns)
            elif strategy == 'fill':
                if columns:
                    cleaned_df[columns] = target_df.fillna(fill_value)
                else:
                    cleaned_df = cleaned_df.fillna(fill_value)
            elif strategy == 'mean':
                numeric_cols = target_df.select_dtypes(include=['number']).columns
                if columns:
                    cleaned_df[numeric_cols] = target_df[numeric_cols].fillna(target_df[numeric_cols].mean())
                else:
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
            elif strategy == 'median':
                numeric_cols = target_df.select_dtypes(include=['number']).columns
                if columns:
                    cleaned_df[numeric_cols] = target_df[numeric_cols].fillna(target_df[numeric_cols].median())
                else:
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
            elif strategy == 'mode':
                if columns:
                    for col in columns:
                        mode_val = target_df[col].mode()
                        if len(mode_val) > 0:
                            cleaned_df[col] = target_df[col].fillna(mode_val[0])
                else:
                    for col in cleaned_df.columns:
                        mode_val = cleaned_df[col].mode()
                        if len(mode_val) > 0:
                            cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
            elif strategy == 'forward_fill':
                if columns:
                    cleaned_df[columns] = target_df.fillna(method='ffill')
                else:
                    cleaned_df = cleaned_df.fillna(method='ffill')
            elif strategy == 'backward_fill':
                if columns:
                    cleaned_df[columns] = target_df.fillna(method='bfill')
                else:
                    cleaned_df = cleaned_df.fillna(method='bfill')
            else:
                return self.create_result(False, message=f"Invalid strategy: {strategy}")
            
            missing_after = cleaned_df.isnull().sum().sum()
            rows_removed = initial_rows - len(cleaned_df)
            
            self.log_operation('handle_missing_values', {
                'strategy': strategy,
                'missing_before': missing_before,
                'missing_after': missing_after,
                'rows_removed': rows_removed
            })
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=f"Handled missing values using '{strategy}' strategy. Removed {rows_removed} rows, {missing_before - missing_after} missing values filled",
                metadata={
                    'strategy': strategy,
                    'missing_before': missing_before,
                    'missing_after': missing_after,
                    'rows_removed': rows_removed
                }
            )
        
        except Exception as e:
            return self.handle_error('handle_missing_values', e)
    
    def standardize_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Rename columns to standard format (lowercase_with_underscores).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            cleaned_df = df.copy()
            old_columns = cleaned_df.columns.tolist()
            new_columns = [standardize_column_name(col) for col in old_columns]
            
            # Handle duplicate column names
            seen = {}
            for i, col in enumerate(new_columns):
                if col in seen:
                    seen[col] += 1
                    new_columns[i] = f"{col}_{seen[col]}"
                else:
                    seen[col] = 0
            
            cleaned_df.columns = new_columns
            
            column_mapping = dict(zip(old_columns, new_columns))
            
            self.log_operation('standardize_columns', {'column_mapping': column_mapping})
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=f"Standardized {len(old_columns)} column names",
                metadata={'column_mapping': column_mapping}
            )
        
        except Exception as e:
            return self.handle_error('standardize_columns', e)
    
    def remove_empty_rows(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Remove rows that are completely empty.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            initial_rows = len(df)
            cleaned_df = df.dropna(how='all')
            removed_rows = initial_rows - len(cleaned_df)
            
            self.log_operation('remove_empty_rows', {'removed_rows': removed_rows})
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=f"Removed {removed_rows} completely empty rows",
                metadata={'removed_rows': removed_rows}
            )
        
        except Exception as e:
            return self.handle_error('remove_empty_rows', e)
    
    def remove_empty_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Remove columns that are completely empty.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Result dictionary with cleaned DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            initial_cols = len(df.columns)
            cleaned_df = df.dropna(axis=1, how='all')
            removed_cols = initial_cols - len(cleaned_df.columns)
            
            self.log_operation('remove_empty_columns', {'removed_columns': removed_cols})
            
            return self.create_result(
                success=True,
                data=cleaned_df,
                message=f"Removed {removed_cols} completely empty columns",
                metadata={'removed_columns': removed_cols}
            )
        
        except Exception as e:
            return self.handle_error('remove_empty_columns', e)
