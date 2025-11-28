"""
Data Analysis Agent - handles statistical analysis and comparison operations.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from .base_agent import BaseAgent
from fuzzywuzzy import fuzz


class AnalysisAgent(BaseAgent):
    """Agent for data analysis operations."""
    
    def __init__(self):
        super().__init__("AnalysisAgent")
    
    def get_available_operations(self) -> Dict[str, str]:
        """Get available analysis operations."""
        return {
            'statistical_summary': 'Generate statistical summary of data',
            'group_analysis': 'Group by columns and aggregate',
            'correlation_analysis': 'Calculate correlation matrix',
            'detect_duplicates_across_files': 'Find matching records across files',
            'compare_datasets': 'Find differences and overlaps between datasets',
            'value_counts': 'Count unique values in columns',
            'cross_tabulation': 'Create cross-tabulation of two columns'
        }
    
    def statistical_summary(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to analyze (None = all columns)
            
        Returns:
            Result dictionary with summary statistics
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            if columns:
                target_df = df[columns]
            else:
                target_df = df
            
            summary = {}
            
            # Numeric columns
            numeric_cols = target_df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                summary['numeric'] = target_df[numeric_cols].describe().to_dict()
                
                # Add additional stats
                for col in numeric_cols:
                    summary['numeric'][col]['median'] = target_df[col].median()
                    summary['numeric'][col]['mode'] = target_df[col].mode().iloc[0] if len(target_df[col].mode()) > 0 else None
                    summary['numeric'][col]['missing'] = target_df[col].isnull().sum()
            
            # Categorical columns
            categorical_cols = target_df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                summary['categorical'] = {}
                for col in categorical_cols:
                    summary['categorical'][col] = {
                        'unique_values': target_df[col].nunique(),
                        'most_common': target_df[col].mode().iloc[0] if len(target_df[col].mode()) > 0 else None,
                        'missing': target_df[col].isnull().sum(),
                        'top_5_values': target_df[col].value_counts().head(5).to_dict()
                    }
            
            # Overall stats
            summary['overall'] = {
                'total_rows': len(target_df),
                'total_columns': len(target_df.columns),
                'total_missing': target_df.isnull().sum().sum(),
                'duplicate_rows': target_df.duplicated().sum(),
                'memory_usage_mb': target_df.memory_usage(deep=True).sum() / (1024 * 1024)
            }
            
            self.log_operation('statistical_summary', {
                'columns_analyzed': len(target_df.columns)
            })
            
            # Create a summary DataFrame for display
            summary_df = pd.DataFrame({
                'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Duplicate Rows', 'Memory (MB)'],
                'Value': [
                    summary['overall']['total_rows'],
                    summary['overall']['total_columns'],
                    summary['overall']['total_missing'],
                    summary['overall']['duplicate_rows'],
                    round(summary['overall']['memory_usage_mb'], 2)
                ]
            })
            
            return self.create_result(
                success=True,
                data=summary_df,
                message=f"Generated statistical summary for {len(target_df.columns)} columns",
                metadata={'detailed_summary': summary}
            )
        
        except Exception as e:
            return self.handle_error('statistical_summary', e)
    
    def group_analysis(
        self,
        df: pd.DataFrame,
        group_by: List[str],
        agg_columns: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Group by columns and aggregate.
        
        Args:
            df: Input DataFrame
            group_by: Columns to group by
            agg_columns: Dictionary of column: aggregation_function
                        Functions: 'sum', 'mean', 'median', 'count', 'min', 'max', 'std'
            
        Returns:
            Result dictionary with grouped data
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            for col in group_by:
                exists, error = self.validate_column_exists(df, col)
                if not exists:
                    return self.create_result(False, message=error)
            
            for col in agg_columns.keys():
                exists, error = self.validate_column_exists(df, col)
                if not exists:
                    return self.create_result(False, message=error)
            
            grouped_df = df.groupby(group_by).agg(agg_columns).reset_index()
            
            self.log_operation('group_analysis', {
                'group_by': group_by,
                'aggregations': agg_columns,
                'result_rows': len(grouped_df)
            })
            
            return self.create_result(
                success=True,
                data=grouped_df,
                message=f"Grouped by {', '.join(group_by)} with {len(agg_columns)} aggregations",
                metadata={'group_by': group_by, 'aggregations': agg_columns}
            )
        
        except Exception as e:
            return self.handle_error('group_analysis', e)
    
    def correlation_analysis(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """
        Calculate correlation matrix.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to analyze (None = all numeric columns)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Result dictionary with correlation matrix
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            if columns:
                numeric_df = df[columns]
            else:
                numeric_df = df.select_dtypes(include=['number'])
            
            if numeric_df.empty or len(numeric_df.columns) < 2:
                return self.create_result(
                    False,
                    message="Need at least 2 numeric columns for correlation analysis"
                )
            
            corr_matrix = numeric_df.corr(method=method)
            
            self.log_operation('correlation_analysis', {
                'columns': len(numeric_df.columns),
                'method': method
            })
            
            return self.create_result(
                success=True,
                data=corr_matrix,
                message=f"Calculated {method} correlation for {len(numeric_df.columns)} columns",
                metadata={'method': method, 'columns': numeric_df.columns.tolist()}
            )
        
        except Exception as e:
            return self.handle_error('correlation_analysis', e)
    
    def detect_duplicates_across_files(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        column: str,
        fuzzy_match: bool = False,
        threshold: int = 80
    ) -> Dict[str, Any]:
        """
        Find matching records across two files.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            column: Column to compare
            fuzzy_match: Use fuzzy string matching
            threshold: Similarity threshold for fuzzy matching (0-100)
            
        Returns:
            Result dictionary with matching records
        """
        try:
            is_valid, error = self.validate_dataframe(df1, "df1")
            if not is_valid:
                return self.create_result(False, message=error)
            
            is_valid, error = self.validate_dataframe(df2, "df2")
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df1, column)
            if not exists:
                return self.create_result(False, message=f"df1: {error}")
            
            exists, error = self.validate_column_exists(df2, column)
            if not exists:
                return self.create_result(False, message=f"df2: {error}")
            
            if fuzzy_match:
                matches = []
                for val1 in df1[column].dropna().unique():
                    for val2 in df2[column].dropna().unique():
                        similarity = fuzz.ratio(str(val1), str(val2))
                        if similarity >= threshold:
                            matches.append({
                                'value_df1': val1,
                                'value_df2': val2,
                                'similarity': similarity
                            })
                
                matches_df = pd.DataFrame(matches)
            else:
                # Exact match
                common_values = set(df1[column].dropna()) & set(df2[column].dropna())
                matches_df = pd.DataFrame({
                    'common_value': list(common_values),
                    'count_df1': [df1[df1[column] == val].shape[0] for val in common_values],
                    'count_df2': [df2[df2[column] == val].shape[0] for val in common_values]
                })
            
            self.log_operation('detect_duplicates_across_files', {
                'column': column,
                'fuzzy_match': fuzzy_match,
                'matches_found': len(matches_df)
            })
            
            return self.create_result(
                success=True,
                data=matches_df,
                message=f"Found {len(matches_df)} matching records in column '{column}'",
                metadata={
                    'column': column,
                    'fuzzy_match': fuzzy_match,
                    'matches_found': len(matches_df)
                }
            )
        
        except Exception as e:
            return self.handle_error('detect_duplicates_across_files', e)
    
    def compare_datasets(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        key_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two datasets to find differences and overlaps.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
            key_column: Column to use as key for comparison
            
        Returns:
            Result dictionary with comparison results
        """
        try:
            is_valid, error = self.validate_dataframe(df1, "df1")
            if not is_valid:
                return self.create_result(False, message=error)
            
            is_valid, error = self.validate_dataframe(df2, "df2")
            if not is_valid:
                return self.create_result(False, message=error)
            
            comparison = {
                'df1_rows': len(df1),
                'df2_rows': len(df2),
                'df1_columns': len(df1.columns),
                'df2_columns': len(df2.columns),
                'common_columns': list(set(df1.columns) & set(df2.columns)),
                'df1_only_columns': list(set(df1.columns) - set(df2.columns)),
                'df2_only_columns': list(set(df2.columns) - set(df1.columns))
            }
            
            if key_column:
                if key_column in df1.columns and key_column in df2.columns:
                    keys1 = set(df1[key_column].dropna())
                    keys2 = set(df2[key_column].dropna())
                    
                    comparison['common_keys'] = len(keys1 & keys2)
                    comparison['df1_only_keys'] = len(keys1 - keys2)
                    comparison['df2_only_keys'] = len(keys2 - keys1)
            
            # Create summary DataFrame
            summary_df = pd.DataFrame({
                'Metric': [
                    'Rows in DF1',
                    'Rows in DF2',
                    'Columns in DF1',
                    'Columns in DF2',
                    'Common Columns',
                    'DF1 Only Columns',
                    'DF2 Only Columns'
                ],
                'Value': [
                    comparison['df1_rows'],
                    comparison['df2_rows'],
                    comparison['df1_columns'],
                    comparison['df2_columns'],
                    len(comparison['common_columns']),
                    len(comparison['df1_only_columns']),
                    len(comparison['df2_only_columns'])
                ]
            })
            
            self.log_operation('compare_datasets', comparison)
            
            return self.create_result(
                success=True,
                data=summary_df,
                message="Dataset comparison completed",
                metadata={'detailed_comparison': comparison}
            )
        
        except Exception as e:
            return self.handle_error('compare_datasets', e)
    
    def value_counts(
        self,
        df: pd.DataFrame,
        column: str,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Count unique values in a column.
        
        Args:
            df: Input DataFrame
            column: Column to analyze
            top_n: Number of top values to return
            
        Returns:
            Result dictionary with value counts
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, column)
            if not exists:
                return self.create_result(False, message=error)
            
            value_counts = df[column].value_counts().head(top_n)
            result_df = pd.DataFrame({
                'Value': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(df) * 100).round(2)
            })
            
            self.log_operation('value_counts', {
                'column': column,
                'unique_values': df[column].nunique(),
                'top_n': top_n
            })
            
            return self.create_result(
                success=True,
                data=result_df,
                message=f"Top {top_n} values in column '{column}'",
                metadata={
                    'column': column,
                    'total_unique': df[column].nunique(),
                    'total_rows': len(df)
                }
            )
        
        except Exception as e:
            return self.handle_error('value_counts', e)
    
    def cross_tabulation(
        self,
        df: pd.DataFrame,
        row_column: str,
        col_column: str,
        normalize: bool = False
    ) -> Dict[str, Any]:
        """
        Create cross-tabulation of two columns.
        
        Args:
            df: Input DataFrame
            row_column: Column for rows
            col_column: Column for columns
            normalize: If True, show percentages instead of counts
            
        Returns:
            Result dictionary with cross-tabulation
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, row_column)
            if not exists:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, col_column)
            if not exists:
                return self.create_result(False, message=error)
            
            crosstab = pd.crosstab(
                df[row_column],
                df[col_column],
                normalize='all' if normalize else False
            )
            
            if normalize:
                crosstab = (crosstab * 100).round(2)
            
            self.log_operation('cross_tabulation', {
                'row_column': row_column,
                'col_column': col_column,
                'normalize': normalize
            })
            
            return self.create_result(
                success=True,
                data=crosstab,
                message=f"Cross-tabulation of '{row_column}' vs '{col_column}'",
                metadata={
                    'row_column': row_column,
                    'col_column': col_column,
                    'normalized': normalize
                }
            )
        
        except Exception as e:
            return self.handle_error('cross_tabulation', e)
