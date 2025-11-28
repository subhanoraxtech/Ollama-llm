"""
Data Transformation Agent - handles data transformation operations.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from .base_agent import BaseAgent


class TransformationAgent(BaseAgent):
    """Agent for data transformation operations."""
    
    def __init__(self):
        super().__init__("TransformationAgent")
    
    def get_available_operations(self) -> Dict[str, str]:
        """Get available transformation operations."""
        return {
            'merge_csvs': 'Merge/join two CSV files',
            'filter_data': 'Filter rows based on conditions',
            'sort_data': 'Sort data by columns',
            'add_column': 'Add a new computed column',
            'combine_columns': 'Combine multiple columns into one',
            'split_column': 'Split a column into multiple columns',
            'rename_columns': 'Rename specific columns',
            'select_columns': 'Select specific columns',
            'drop_columns': 'Drop specific columns',
            'pivot_data': 'Pivot table transformation',
            'melt_data': 'Unpivot table transformation'
        }
    
    def merge_csvs(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        how: str = 'inner',
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        suffixes: tuple = ('_left', '_right')
    ) -> Dict[str, Any]:
        """
        Merge two DataFrames.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            how: Type of merge ('inner', 'left', 'right', 'outer')
            on: Column(s) to join on (must exist in both DataFrames)
            left_on: Column(s) from left DataFrame
            right_on: Column(s) from right DataFrame
            suffixes: Suffixes for overlapping columns
            
        Returns:
            Result dictionary with merged DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df1, "df1")
            if not is_valid:
                return self.create_result(False, message=error)
            
            is_valid, error = self.validate_dataframe(df2, "df2")
            if not is_valid:
                return self.create_result(False, message=error)
            
            if how not in ['inner', 'left', 'right', 'outer']:
                return self.create_result(False, message=f"Invalid merge type: {how}")
            
            merged_df = pd.merge(
                df1, df2,
                how=how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                suffixes=suffixes
            )
            
            self.log_operation('merge_csvs', {
                'df1_rows': len(df1),
                'df2_rows': len(df2),
                'merged_rows': len(merged_df),
                'how': how,
                'on': on or (left_on, right_on)
            })
            
            return self.create_result(
                success=True,
                data=merged_df,
                message=f"Merged DataFrames using '{how}' join. Result: {len(merged_df)} rows",
                metadata={
                    'df1_rows': len(df1),
                    'df2_rows': len(df2),
                    'merged_rows': len(merged_df),
                    'merge_type': how
                }
            )
        
        except Exception as e:
            return self.handle_error('merge_csvs', e)
    
    def filter_data(
        self,
        df: pd.DataFrame,
        conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter rows based on conditions.
        
        Args:
            df: Input DataFrame
            conditions: Dictionary of column: value or column: (operator, value)
                       Operators: '==', '!=', '>', '<', '>=', '<=', 'contains', 'startswith', 'endswith'
            
        Returns:
            Result dictionary with filtered DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            filtered_df = df.copy()
            
            for column, condition in conditions.items():
                exists, error = self.validate_column_exists(filtered_df, column)
                if not exists:
                    return self.create_result(False, message=error)
                
                # Simple value comparison (equals)
                if not isinstance(condition, tuple):
                    filtered_df = filtered_df[filtered_df[column] == condition]
                else:
                    operator, value = condition
                    
                    if operator == '==':
                        filtered_df = filtered_df[filtered_df[column] == value]
                    elif operator == '!=':
                        filtered_df = filtered_df[filtered_df[column] != value]
                    elif operator == '>':
                        filtered_df = filtered_df[filtered_df[column] > value]
                    elif operator == '<':
                        filtered_df = filtered_df[filtered_df[column] < value]
                    elif operator == '>=':
                        filtered_df = filtered_df[filtered_df[column] >= value]
                    elif operator == '<=':
                        filtered_df = filtered_df[filtered_df[column] <= value]
                    elif operator == 'contains':
                        filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), na=False)]
                    elif operator == 'startswith':
                        filtered_df = filtered_df[filtered_df[column].astype(str).str.startswith(str(value), na=False)]
                    elif operator == 'endswith':
                        filtered_df = filtered_df[filtered_df[column].astype(str).str.endswith(str(value), na=False)]
                    else:
                        return self.create_result(False, message=f"Invalid operator: {operator}")
            
            initial_rows = len(df)
            filtered_rows = len(filtered_df)
            
            self.log_operation('filter_data', {
                'conditions': conditions,
                'initial_rows': initial_rows,
                'filtered_rows': filtered_rows
            })
            
            return self.create_result(
                success=True,
                data=filtered_df,
                message=f"Filtered data: {filtered_rows} rows remaining (from {initial_rows})",
                metadata={
                    'initial_rows': initial_rows,
                    'filtered_rows': filtered_rows,
                    'conditions': conditions
                }
            )
        
        except Exception as e:
            return self.handle_error('filter_data', e)
    
    def sort_data(
        self,
        df: pd.DataFrame,
        by: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True
    ) -> Dict[str, Any]:
        """
        Sort data by columns.
        
        Args:
            df: Input DataFrame
            by: Column(s) to sort by
            ascending: Sort order(s)
            
        Returns:
            Result dictionary with sorted DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            if isinstance(by, str):
                by = [by]
            
            for column in by:
                exists, error = self.validate_column_exists(df, column)
                if not exists:
                    return self.create_result(False, message=error)
            
            sorted_df = df.sort_values(by=by, ascending=ascending)
            
            self.log_operation('sort_data', {'by': by, 'ascending': ascending})
            
            return self.create_result(
                success=True,
                data=sorted_df,
                message=f"Sorted data by {', '.join(by)}",
                metadata={'sort_by': by, 'ascending': ascending}
            )
        
        except Exception as e:
            return self.handle_error('sort_data', e)
    
    def add_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        value: Any = None,
        formula: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a new column with a constant value or formula.
        
        Args:
            df: Input DataFrame
            column_name: Name of new column
            value: Constant value for the column
            formula: Python expression to evaluate (e.g., "df['col1'] + df['col2']")
            
        Returns:
            Result dictionary with updated DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            result_df = df.copy()
            
            if formula:
                # Evaluate formula (be careful with user input in production!)
                result_df[column_name] = eval(formula, {'df': result_df, 'pd': pd, 'np': np})
            else:
                result_df[column_name] = value
            
            self.log_operation('add_column', {
                'column_name': column_name,
                'has_formula': formula is not None
            })
            
            return self.create_result(
                success=True,
                data=result_df,
                message=f"Added column '{column_name}'",
                metadata={'column_name': column_name}
            )
        
        except Exception as e:
            return self.handle_error('add_column', e)
    
    def combine_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        new_column: str,
        separator: str = ' '
    ) -> Dict[str, Any]:
        """
        Combine multiple columns into one.
        
        Args:
            df: Input DataFrame
            columns: Columns to combine
            new_column: Name of the new combined column
            separator: Separator between values
            
        Returns:
            Result dictionary with updated DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            for column in columns:
                exists, error = self.validate_column_exists(df, column)
                if not exists:
                    return self.create_result(False, message=error)
            
            result_df = df.copy()
            result_df[new_column] = result_df[columns].astype(str).agg(separator.join, axis=1)
            
            self.log_operation('combine_columns', {
                'source_columns': columns,
                'new_column': new_column,
                'separator': separator
            })
            
            return self.create_result(
                success=True,
                data=result_df,
                message=f"Combined {len(columns)} columns into '{new_column}'",
                metadata={'source_columns': columns, 'new_column': new_column}
            )
        
        except Exception as e:
            return self.handle_error('combine_columns', e)
    
    def split_column(
        self,
        df: pd.DataFrame,
        column: str,
        separator: str,
        new_columns: List[str]
    ) -> Dict[str, Any]:
        """
        Split a column into multiple columns.
        
        Args:
            df: Input DataFrame
            column: Column to split
            separator: Separator to split on
            new_columns: Names for the new columns
            
        Returns:
            Result dictionary with updated DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, column)
            if not exists:
                return self.create_result(False, message=error)
            
            result_df = df.copy()
            split_data = result_df[column].astype(str).str.split(separator, expand=True)
            
            # Assign to new columns
            for i, new_col in enumerate(new_columns):
                if i < len(split_data.columns):
                    result_df[new_col] = split_data[i]
            
            self.log_operation('split_column', {
                'source_column': column,
                'new_columns': new_columns,
                'separator': separator
            })
            
            return self.create_result(
                success=True,
                data=result_df,
                message=f"Split column '{column}' into {len(new_columns)} columns",
                metadata={'source_column': column, 'new_columns': new_columns}
            )
        
        except Exception as e:
            return self.handle_error('split_column', e)
    
    def rename_columns(
        self,
        df: pd.DataFrame,
        column_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Rename specific columns.
        
        Args:
            df: Input DataFrame
            column_mapping: Dictionary mapping old names to new names
            
        Returns:
            Result dictionary with updated DataFrame
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            for old_name in column_mapping.keys():
                exists, error = self.validate_column_exists(df, old_name)
                if not exists:
                    return self.create_result(False, message=error)
            
            result_df = df.rename(columns=column_mapping)
            
            self.log_operation('rename_columns', {'column_mapping': column_mapping})
            
            return self.create_result(
                success=True,
                data=result_df,
                message=f"Renamed {len(column_mapping)} columns",
                metadata={'column_mapping': column_mapping}
            )
        
        except Exception as e:
            return self.handle_error('rename_columns', e)
    
    def select_columns(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """
        Select specific columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to select
            
        Returns:
            Result dictionary with selected columns
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            for column in columns:
                exists, error = self.validate_column_exists(df, column)
                if not exists:
                    return self.create_result(False, message=error)
            
            result_df = df[columns].copy()
            
            self.log_operation('select_columns', {'columns': columns})
            
            return self.create_result(
                success=True,
                data=result_df,
                message=f"Selected {len(columns)} columns",
                metadata={'selected_columns': columns}
            )
        
        except Exception as e:
            return self.handle_error('select_columns', e)
    
    def drop_columns(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """
        Drop specific columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to drop
            
        Returns:
            Result dictionary with remaining columns
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            for column in columns:
                exists, error = self.validate_column_exists(df, column)
                if not exists:
                    return self.create_result(False, message=error)
            
            result_df = df.drop(columns=columns)
            
            self.log_operation('drop_columns', {'columns': columns})
            
            return self.create_result(
                success=True,
                data=result_df,
                message=f"Dropped {len(columns)} columns",
                metadata={'dropped_columns': columns}
            )
        
        except Exception as e:
            return self.handle_error('drop_columns', e)
