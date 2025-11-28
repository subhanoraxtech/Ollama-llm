"""
Visualization Agent - handles chart and plot generation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List
import io
import base64
from .base_agent import BaseAgent


class VisualizationAgent(BaseAgent):
    """Agent for data visualization operations."""
    
    def __init__(self):
        super().__init__("VisualizationAgent")
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def get_available_operations(self) -> Dict[str, str]:
        """Get available visualization operations."""
        return {
            'create_bar_chart': 'Create bar chart for categorical data',
            'create_pie_chart': 'Create pie chart for distributions',
            'create_histogram': 'Create histogram for numeric data',
            'create_line_chart': 'Create line chart for time series',
            'create_scatter_plot': 'Create scatter plot for two variables',
            'create_heatmap': 'Create correlation heatmap',
            'create_box_plot': 'Create box plot for distributions',
            'create_summary_table': 'Create formatted summary table'
        }
    
    def create_bar_chart(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_column: Optional[str] = None,
        title: str = "Bar Chart",
        top_n: int = 10,
        interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Create a bar chart.
        
        Args:
            df: Input DataFrame
            x_column: Column for x-axis (categorical)
            y_column: Column for y-axis (if None, will count occurrences)
            title: Chart title
            top_n: Show only top N categories
            interactive: Use plotly (True) or matplotlib (False)
            
        Returns:
            Result dictionary with chart
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, x_column)
            if not exists:
                return self.create_result(False, message=error)
            
            if y_column:
                exists, error = self.validate_column_exists(df, y_column)
                if not exists:
                    return self.create_result(False, message=error)
                
                # Group and aggregate
                plot_data = df.groupby(x_column)[y_column].sum().sort_values(ascending=False).head(top_n)
            else:
                # Count occurrences
                plot_data = df[x_column].value_counts().head(top_n)
            
            if interactive:
                fig = px.bar(
                    x=plot_data.index,
                    y=plot_data.values,
                    title=title,
                    labels={'x': x_column, 'y': y_column or 'Count'}
                )
                chart_html = fig.to_html()
                chart_data = {'type': 'plotly', 'html': chart_html}
            else:
                fig, ax = plt.subplots()
                plot_data.plot(kind='bar', ax=ax)
                ax.set_title(title)
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column or 'Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                chart_data = {'type': 'matplotlib', 'image_base64': img_base64}
            
            self.log_operation('create_bar_chart', {
                'x_column': x_column,
                'y_column': y_column,
                'top_n': top_n
            })
            
            return self.create_result(
                success=True,
                data=chart_data,
                message=f"Created bar chart for '{x_column}'",
                metadata={'chart_type': 'bar', 'x_column': x_column, 'y_column': y_column}
            )
        
        except Exception as e:
            return self.handle_error('create_bar_chart', e)
    
    def create_pie_chart(
        self,
        df: pd.DataFrame,
        column: str,
        title: str = "Pie Chart",
        top_n: int = 10,
        interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Create a pie chart.
        
        Args:
            df: Input DataFrame
            column: Column to visualize
            title: Chart title
            top_n: Show only top N categories
            interactive: Use plotly (True) or matplotlib (False)
            
        Returns:
            Result dictionary with chart
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, column)
            if not exists:
                return self.create_result(False, message=error)
            
            value_counts = df[column].value_counts().head(top_n)
            
            if interactive:
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=title
                )
                chart_html = fig.to_html()
                chart_data = {'type': 'plotly', 'html': chart_html}
            else:
                fig, ax = plt.subplots()
                ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                ax.set_title(title)
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                chart_data = {'type': 'matplotlib', 'image_base64': img_base64}
            
            self.log_operation('create_pie_chart', {'column': column, 'top_n': top_n})
            
            return self.create_result(
                success=True,
                data=chart_data,
                message=f"Created pie chart for '{column}'",
                metadata={'chart_type': 'pie', 'column': column}
            )
        
        except Exception as e:
            return self.handle_error('create_pie_chart', e)
    
    def create_histogram(
        self,
        df: pd.DataFrame,
        column: str,
        bins: int = 30,
        title: str = "Histogram",
        interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Create a histogram.
        
        Args:
            df: Input DataFrame
            column: Numeric column to visualize
            bins: Number of bins
            title: Chart title
            interactive: Use plotly (True) or matplotlib (False)
            
        Returns:
            Result dictionary with chart
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, column)
            if not exists:
                return self.create_result(False, message=error)
            
            if interactive:
                fig = px.histogram(df, x=column, nbins=bins, title=title)
                chart_html = fig.to_html()
                chart_data = {'type': 'plotly', 'html': chart_html}
            else:
                fig, ax = plt.subplots()
                df[column].hist(bins=bins, ax=ax)
                ax.set_title(title)
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                chart_data = {'type': 'matplotlib', 'image_base64': img_base64}
            
            self.log_operation('create_histogram', {'column': column, 'bins': bins})
            
            return self.create_result(
                success=True,
                data=chart_data,
                message=f"Created histogram for '{column}'",
                metadata={'chart_type': 'histogram', 'column': column}
            )
        
        except Exception as e:
            return self.handle_error('create_histogram', e)
    
    def create_line_chart(
        self,
        df: pd.DataFrame,
        x_column: str,
        y_columns: List[str],
        title: str = "Line Chart",
        interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Create a line chart.
        
        Args:
            df: Input DataFrame
            x_column: Column for x-axis
            y_columns: Column(s) for y-axis
            title: Chart title
            interactive: Use plotly (True) or matplotlib (False)
            
        Returns:
            Result dictionary with chart
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            exists, error = self.validate_column_exists(df, x_column)
            if not exists:
                return self.create_result(False, message=error)
            
            for col in y_columns:
                exists, error = self.validate_column_exists(df, col)
                if not exists:
                    return self.create_result(False, message=error)
            
            if interactive:
                fig = px.line(df, x=x_column, y=y_columns, title=title)
                chart_html = fig.to_html()
                chart_data = {'type': 'plotly', 'html': chart_html}
            else:
                fig, ax = plt.subplots()
                for col in y_columns:
                    ax.plot(df[x_column], df[col], label=col, marker='o')
                ax.set_title(title)
                ax.set_xlabel(x_column)
                ax.set_ylabel('Value')
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.read()).decode()
                plt.close()
                
                chart_data = {'type': 'matplotlib', 'image_base64': img_base64}
            
            self.log_operation('create_line_chart', {
                'x_column': x_column,
                'y_columns': y_columns
            })
            
            return self.create_result(
                success=True,
                data=chart_data,
                message=f"Created line chart",
                metadata={'chart_type': 'line', 'x_column': x_column, 'y_columns': y_columns}
            )
        
        except Exception as e:
            return self.handle_error('create_line_chart', e)
    
    def create_heatmap(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Heatmap"
    ) -> Dict[str, Any]:
        """
        Create a correlation heatmap.
        
        Args:
            df: Input DataFrame
            columns: Specific columns (None = all numeric columns)
            title: Chart title
            
        Returns:
            Result dictionary with chart
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
                    message="Need at least 2 numeric columns for heatmap"
                )
            
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            chart_data = {'type': 'matplotlib', 'image_base64': img_base64}
            
            self.log_operation('create_heatmap', {'columns': len(numeric_df.columns)})
            
            return self.create_result(
                success=True,
                data=chart_data,
                message=f"Created correlation heatmap for {len(numeric_df.columns)} columns",
                metadata={'chart_type': 'heatmap', 'columns': numeric_df.columns.tolist()}
            )
        
        except Exception as e:
            return self.handle_error('create_heatmap', e)
    
    def create_summary_table(
        self,
        df: pd.DataFrame,
        max_rows: int = 100
    ) -> Dict[str, Any]:
        """
        Create a formatted summary table.
        
        Args:
            df: Input DataFrame
            max_rows: Maximum rows to display
            
        Returns:
            Result dictionary with formatted table
        """
        try:
            is_valid, error = self.validate_dataframe(df)
            if not is_valid:
                return self.create_result(False, message=error)
            
            display_df = df.head(max_rows).copy()
            
            # Format numeric columns
            numeric_cols = display_df.select_dtypes(include=['float']).columns
            for col in numeric_cols:
                display_df[col] = display_df[col].round(2)
            
            self.log_operation('create_summary_table', {'rows': len(display_df)})
            
            return self.create_result(
                success=True,
                data=display_df,
                message=f"Created summary table with {len(display_df)} rows",
                metadata={'total_rows': len(df), 'displayed_rows': len(display_df)}
            )
        
        except Exception as e:
            return self.handle_error('create_summary_table', e)
