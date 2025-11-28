"""
Multi-CSV Comparator Module
Handles comparison, merging, and analysis of multiple CSV files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class CSVComparator:
    """Comprehensive CSV comparison and manipulation"""
    
    def __init__(self):
        pass
    
    def compare_structure(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                         name1: str = "CSV 1", name2: str = "CSV 2") -> Dict:
        """
        Compare structure of two CSVs
        
        Returns:
            dict with columns, types, shape comparison
        """
        result = {
            'success': True,
            'name1': name1,
            'name2': name2,
            'shape1': df1.shape,
            'shape2': df2.shape,
            'columns1': list(df1.columns),
            'columns2': list(df2.columns),
            'common_columns': list(set(df1.columns) & set(df2.columns)),
            'only_in_1': list(set(df1.columns) - set(df2.columns)),
            'only_in_2': list(set(df2.columns) - set(df1.columns)),
            'dtypes1': df1.dtypes.to_dict(),
            'dtypes2': df2.dtypes.to_dict(),
        }
        
        # Compare data types for common columns
        type_mismatches = {}
        for col in result['common_columns']:
            if str(df1[col].dtype) != str(df2[col].dtype):
                type_mismatches[col] = {
                    name1: str(df1[col].dtype),
                    name2: str(df2[col].dtype)
                }
        
        result['type_mismatches'] = type_mismatches
        result['structure_identical'] = (
            result['shape1'] == result['shape2'] and
            len(result['only_in_1']) == 0 and
            len(result['only_in_2']) == 0 and
            len(type_mismatches) == 0
        )
        
        return result
    
    def compare_data(self, df1: pd.DataFrame, df2: pd.DataFrame,
                    key_column: Optional[str] = None,
                    name1: str = "CSV 1", name2: str = "CSV 2") -> Dict:
        """
        Compare data between two CSVs
        
        Args:
            key_column: Column to use as unique identifier
        """
        result = {
            'success': True,
            'name1': name1,
            'name2': name2,
        }
        
        # Find common columns
        common_cols = list(set(df1.columns) & set(df2.columns))
        
        if not common_cols:
            result['success'] = False
            result['error'] = "No common columns found"
            return result
        
        # If key column specified, use it for matching
        if key_column and key_column in common_cols:
            # Match rows by key
            merged = pd.merge(
                df1[[key_column] + [c for c in common_cols if c != key_column]],
                df2[[key_column] + [c for c in common_cols if c != key_column]],
                on=key_column,
                how='outer',
                suffixes=('_1', '_2'),
                indicator=True
            )
            
            result['total_rows_1'] = len(df1)
            result['total_rows_2'] = len(df2)
            result['rows_only_in_1'] = len(merged[merged['_merge'] == 'left_only'])
            result['rows_only_in_2'] = len(merged[merged['_merge'] == 'right_only'])
            result['common_rows'] = len(merged[merged['_merge'] == 'both'])
            
            # Find value mismatches in common rows
            mismatches = []
            common_data = merged[merged['_merge'] == 'both']
            
            for col in common_cols:
                if col == key_column:
                    continue
                col1 = f"{col}_1"
                col2 = f"{col}_2"
                if col1 in common_data.columns and col2 in common_data.columns:
                    diff_mask = common_data[col1] != common_data[col2]
                    if diff_mask.any():
                        mismatches.append({
                            'column': col,
                            'mismatch_count': diff_mask.sum(),
                            'examples': common_data[diff_mask][[key_column, col1, col2]].head(5).to_dict('records')
                        })
            
            result['value_mismatches'] = mismatches
            result['data_identical'] = len(mismatches) == 0
            
        else:
            # Simple row-by-row comparison (no key)
            result['total_rows_1'] = len(df1)
            result['total_rows_2'] = len(df2)
            result['rows_match'] = result['total_rows_1'] == result['total_rows_2']
            
            if result['rows_match']:
                # Compare common columns
                mismatches = []
                for col in common_cols:
                    if not df1[col].equals(df2[col]):
                        diff_count = (df1[col] != df2[col]).sum()
                        mismatches.append({
                            'column': col,
                            'mismatch_count': diff_count
                        })
                
                result['value_mismatches'] = mismatches
                result['data_identical'] = len(mismatches) == 0
            else:
                result['data_identical'] = False
        
        return result
    
    def merge_csvs(self, df1: pd.DataFrame, df2: pd.DataFrame,
                   join_type: str = 'inner', on_column: Optional[str] = None,
                   left_on: Optional[str] = None, right_on: Optional[str] = None) -> Dict:
        """
        Merge two CSVs with SQL-style joins
        
        Args:
            join_type: 'inner', 'left', 'right', 'outer', 'cross'
            on_column: Column name to join on (if same in both)
            left_on, right_on: Different column names to join on
        """
        try:
            if join_type == 'cross':
                # Cross join (cartesian product)
                df1['_key'] = 1
                df2['_key'] = 1
                result_df = pd.merge(df1, df2, on='_key', suffixes=('_1', '_2')).drop('_key', axis=1)
            else:
                # Regular joins
                if on_column:
                    result_df = pd.merge(df1, df2, on=on_column, how=join_type, suffixes=('_1', '_2'))
                elif left_on and right_on:
                    result_df = pd.merge(df1, df2, left_on=left_on, right_on=right_on, 
                                       how=join_type, suffixes=('_1', '_2'))
                else:
                    # Try to find common columns
                    common = list(set(df1.columns) & set(df2.columns))
                    if common:
                        result_df = pd.merge(df1, df2, on=common[0], how=join_type, suffixes=('_1', '_2'))
                    else:
                        return {
                            'success': False,
                            'error': 'No common columns found and no join column specified'
                        }
            
            return {
                'success': True,
                'data': result_df,
                'rows': len(result_df),
                'columns': len(result_df.columns),
                'join_type': join_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def find_differences(self, df1: pd.DataFrame, df2: pd.DataFrame,
                        key_column: str, name1: str = "CSV 1", name2: str = "CSV 2") -> Dict:
        """
        Find new, deleted, and modified rows between two CSVs
        """
        try:
            # Merge with indicator
            merged = pd.merge(df1, df2, on=key_column, how='outer', 
                            suffixes=('_old', '_new'), indicator=True)
            
            # New rows (only in df2)
            new_rows = merged[merged['_merge'] == 'right_only'].copy()
            new_rows = new_rows[[c for c in new_rows.columns if not c.endswith('_old') and c != '_merge']]
            new_rows.columns = [c.replace('_new', '') for c in new_rows.columns]
            
            # Deleted rows (only in df1)
            deleted_rows = merged[merged['_merge'] == 'left_only'].copy()
            deleted_rows = deleted_rows[[c for c in deleted_rows.columns if not c.endswith('_new') and c != '_merge']]
            deleted_rows.columns = [c.replace('_old', '') for c in deleted_rows.columns]
            
            # Modified rows (in both but different)
            common_rows = merged[merged['_merge'] == 'both']
            modified_rows = []
            
            for idx, row in common_rows.iterrows():
                changes = {}
                for col in df1.columns:
                    if col == key_column:
                        continue
                    old_col = f"{col}_old"
                    new_col = f"{col}_new"
                    if old_col in row and new_col in row:
                        if pd.notna(row[old_col]) or pd.notna(row[new_col]):
                            if row[old_col] != row[new_col]:
                                changes[col] = {
                                    'old': row[old_col],
                                    'new': row[new_col]
                                }
                
                if changes:
                    modified_rows.append({
                        key_column: row[key_column],
                        'changes': changes
                    })
            
            return {
                'success': True,
                'new_rows': new_rows,
                'deleted_rows': deleted_rows,
                'modified_rows': modified_rows,
                'summary': {
                    'new_count': len(new_rows),
                    'deleted_count': len(deleted_rows),
                    'modified_count': len(modified_rows),
                    'unchanged_count': len(common_rows) - len(modified_rows)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive statistics for a CSV"""
        stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
        }
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns statistics
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            stats['categorical_stats'] = {}
            for col in cat_cols[:10]:  # Limit to first 10
                stats['categorical_stats'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
        
        return stats
