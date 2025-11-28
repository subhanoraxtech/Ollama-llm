"""
Advanced Analytics Agent - handles ML-based analytics and validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.validators import validate_email, validate_phone


class AdvancedAnalyticsAgent(BaseAgent):
    """Agent for advanced analytics operations."""
    
    def __init__(self):
        super().__init__("AdvancedAnalyticsAgent")
    
    def get_available_operations(self) -> Dict[str, str]:
        """Get available advanced analytics operations."""
        return {
            'customer_segmentation': 'Segment customers using K-means clustering',
            'detect_outliers': 'Detect statistical outliers in numeric columns',
            'validate_emails': 'Validate all email addresses in a column',
            'validate_phones': 'Validate all phone numbers in a column',
            'churn_prediction': 'Basic churn likelihood scoring',
            'identify_high_value_customers': 'Identify high-value customers'
        }
    
    def customer_segmentation(
        self,
        df: pd.DataFrame,
        features: List[str],
        n_clusters: int = 3,
        cluster_column_name: str = 'segment'
    ) -> Dict[str, Any]:
        """
        Segment customers using K-means clustering.
        
        Args:
            df: Input DataFrame
            features: Columns to use for clustering (must be numeric)
            n_clusters: Number of clusters
            cluster_column_name: Name for the cluster assignment column
            
        Returns:
            Result dictionary with segmented data
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            for feature in features:
                exists, error = self.validate_column_exists(df, feature)
                if not exists:
                    return self.create_result(False, message=error)
            
            # Prepare data
            X = df[features].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster assignments to DataFrame
            result_df = df.copy()
            result_df[cluster_column_name] = clusters
            
            # Calculate cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_data = result_df[result_df[cluster_column_name] == i]
                cluster_stats[f'Cluster {i}'] = {
                    'size': len(cluster_data),
                    'percentage': round(len(cluster_data) / len(result_df) * 100, 2),
                    'mean_values': cluster_data[features].mean().to_dict()
                }
            
            self.log_operation('customer_segmentation', {
                'features': features,
                'n_clusters': n_clusters,
                'cluster_stats': cluster_stats
            })
            
            return self.create_result(
                success=True,
                data=result_df,
                message=f"Created {n_clusters} customer segments",
                metadata={
                    'n_clusters': n_clusters,
                    'features': features,
                    'cluster_stats': cluster_stats
                }
            )
        
        except Exception as e:
            return self.handle_error('customer_segmentation', e)
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check (None = all numeric columns)
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (IQR multiplier or Z-score)
            
        Returns:
            Result dictionary with outlier information
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            if columns is None:
                columns = df.select_dtypes(include=['number']).columns.tolist()
            
            outliers_info = {}
            outlier_indices = set()
            
            for column in columns:
                exists, error = self.validate_column_exists(df, column)
                if not exists:
                    continue
                
                col_data = df[column].dropna()
                
                if method == 'iqr':
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                
                elif method == 'zscore':
                    mean = col_data.mean()
                    std = col_data.std()
                    z_scores = np.abs((df[column] - mean) / std)
                    outlier_mask = z_scores > threshold
                
                else:
                    return self.create_result(False, message=f"Invalid method: {method}")
                
                outlier_count = outlier_mask.sum()
                outlier_indices.update(df[outlier_mask].index.tolist())
                
                outliers_info[column] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_count / len(df) * 100, 2),
                    'values': df[outlier_mask][column].tolist()[:10]  # First 10 outliers
                }
            
            # Create summary DataFrame
            summary_df = pd.DataFrame({
                'Column': list(outliers_info.keys()),
                'Outlier Count': [info['count'] for info in outliers_info.values()],
                'Percentage': [info['percentage'] for info in outliers_info.values()]
            })
            
            self.log_operation('detect_outliers', {
                'method': method,
                'columns': len(columns),
                'total_outlier_rows': len(outlier_indices)
            })
            
            return self.create_result(
                success=True,
                data=summary_df,
                message=f"Detected outliers in {len(columns)} columns using {method} method",
                metadata={
                    'method': method,
                    'threshold': threshold,
                    'detailed_outliers': outliers_info,
                    'total_outlier_rows': len(outlier_indices)
                }
            )
        
        except Exception as e:
            return self.handle_error('detect_outliers', e)
    
    def validate_emails(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Dict[str, Any]:
        """
        Validate email addresses in a column.
        
        Args:
            df: Input DataFrame
            column: Column containing emails
            
        Returns:
            Result dictionary with validation results
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, column)
            if not exists:
                return self.create_result(False, message=error)
            
            result_df = df.copy()
            result_df[f'{column}_valid'] = result_df[column].apply(validate_email)
            
            valid_count = result_df[f'{column}_valid'].sum()
            invalid_count = len(result_df) - valid_count
            
            # Get sample invalid emails
            invalid_emails = result_df[~result_df[f'{column}_valid']][column].head(10).tolist()
            
            summary_df = pd.DataFrame({
                'Status': ['Valid', 'Invalid'],
                'Count': [valid_count, invalid_count],
                'Percentage': [
                    round(valid_count / len(result_df) * 100, 2),
                    round(invalid_count / len(result_df) * 100, 2)
                ]
            })
            
            self.log_operation('validate_emails', {
                'column': column,
                'valid_count': valid_count,
                'invalid_count': invalid_count
            })
            
            return self.create_result(
                success=True,
                data=summary_df,
                message=f"Validated {len(result_df)} emails: {valid_count} valid, {invalid_count} invalid",
                metadata={
                    'column': column,
                    'valid_count': int(valid_count),
                    'invalid_count': int(invalid_count),
                    'sample_invalid': invalid_emails,
                    'result_dataframe': result_df
                }
            )
        
        except Exception as e:
            return self.handle_error('validate_emails', e)
    
    def validate_phones(
        self,
        df: pd.DataFrame,
        column: str
    ) -> Dict[str, Any]:
        """
        Validate phone numbers in a column.
        
        Args:
            df: Input DataFrame
            column: Column containing phone numbers
            
        Returns:
            Result dictionary with validation results
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, column)
            if not exists:
                return self.create_result(False, message=error)
            
            result_df = df.copy()
            result_df[f'{column}_valid'] = result_df[column].apply(validate_phone)
            
            valid_count = result_df[f'{column}_valid'].sum()
            invalid_count = len(result_df) - valid_count
            
            # Get sample invalid phones
            invalid_phones = result_df[~result_df[f'{column}_valid']][column].head(10).tolist()
            
            summary_df = pd.DataFrame({
                'Status': ['Valid', 'Invalid'],
                'Count': [valid_count, invalid_count],
                'Percentage': [
                    round(valid_count / len(result_df) * 100, 2),
                    round(invalid_count / len(result_df) * 100, 2)
                ]
            })
            
            self.log_operation('validate_phones', {
                'column': column,
                'valid_count': valid_count,
                'invalid_count': invalid_count
            })
            
            return self.create_result(
                success=True,
                data=summary_df,
                message=f"Validated {len(result_df)} phone numbers: {valid_count} valid, {invalid_count} invalid",
                metadata={
                    'column': column,
                    'valid_count': int(valid_count),
                    'invalid_count': int(invalid_count),
                    'sample_invalid': invalid_phones,
                    'result_dataframe': result_df
                }
            )
        
        except Exception as e:
            return self.handle_error('validate_phones', e)
    
    def churn_prediction(
        self,
        df: pd.DataFrame,
        activity_column: str,
        threshold_days: int = 30
    ) -> Dict[str, Any]:
        """
        Basic churn likelihood scoring based on activity.
        
        Args:
            df: Input DataFrame
            activity_column: Column containing last activity date
            threshold_days: Days of inactivity to consider as churn risk
            
        Returns:
            Result dictionary with churn scores
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, activity_column)
            if not exists:
                return self.create_result(False, message=error)
            
            result_df = df.copy()
            
            # Convert to datetime
            result_df[activity_column] = pd.to_datetime(result_df[activity_column], errors='coerce')
            
            # Calculate days since last activity
            today = pd.Timestamp.now()
            result_df['days_since_activity'] = (today - result_df[activity_column]).dt.days
            
            # Assign churn risk
            result_df['churn_risk'] = pd.cut(
                result_df['days_since_activity'],
                bins=[-np.inf, threshold_days, threshold_days * 2, np.inf],
                labels=['Low', 'Medium', 'High']
            )
            
            # Count by risk level
            risk_counts = result_df['churn_risk'].value_counts()
            
            summary_df = pd.DataFrame({
                'Risk Level': risk_counts.index,
                'Count': risk_counts.values,
                'Percentage': (risk_counts.values / len(result_df) * 100).round(2)
            })
            
            self.log_operation('churn_prediction', {
                'activity_column': activity_column,
                'threshold_days': threshold_days,
                'risk_distribution': risk_counts.to_dict()
            })
            
            return self.create_result(
                success=True,
                data=summary_df,
                message=f"Calculated churn risk for {len(result_df)} records",
                metadata={
                    'threshold_days': threshold_days,
                    'risk_distribution': risk_counts.to_dict(),
                    'result_dataframe': result_df
                }
            )
        
        except Exception as e:
            return self.handle_error('churn_prediction', e)
    
    def identify_high_value_customers(
        self,
        df: pd.DataFrame,
        value_column: str,
        top_percentage: float = 20.0
    ) -> Dict[str, Any]:
        """
        Identify high-value customers based on a value metric.
        
        Args:
            df: Input DataFrame
            value_column: Column containing customer value (e.g., revenue, purchases)
            top_percentage: Percentage of top customers to identify
            
        Returns:
            Result dictionary with high-value customer data
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, value_column)
            if not exists:
                return self.create_result(False, message=error)
            
            result_df = df.copy()
            
            # Calculate percentile threshold
            threshold = result_df[value_column].quantile(1 - top_percentage / 100)
            
            # Identify high-value customers
            result_df['high_value'] = result_df[value_column] >= threshold
            
            high_value_count = result_df['high_value'].sum()
            total_value = result_df[value_column].sum()
            high_value_total = result_df[result_df['high_value']][value_column].sum()
            
            summary_df = pd.DataFrame({
                'Metric': [
                    'Total Customers',
                    'High-Value Customers',
                    'Percentage',
                    f'Total {value_column}',
                    f'High-Value {value_column}',
                    'Value Concentration %'
                ],
                'Value': [
                    len(result_df),
                    int(high_value_count),
                    f"{round(high_value_count / len(result_df) * 100, 2)}%",
                    round(total_value, 2),
                    round(high_value_total, 2),
                    f"{round(high_value_total / total_value * 100, 2)}%"
                ]
            })
            
            self.log_operation('identify_high_value_customers', {
                'value_column': value_column,
                'top_percentage': top_percentage,
                'high_value_count': int(high_value_count)
            })
            
            return self.create_result(
                success=True,
                data=summary_df,
                message=f"Identified {int(high_value_count)} high-value customers (top {top_percentage}%)",
                metadata={
                    'value_column': value_column,
                    'threshold': threshold,
                    'high_value_count': int(high_value_count),
                    'value_concentration': round(high_value_total / total_value * 100, 2),
                    'result_dataframe': result_df
                }
            )
        
        except Exception as e:
            return self.handle_error('identify_high_value_customers', e)
