"""
CSV Schema Analyzer - Intelligent CSV Structure Analysis
Provides detailed schema information to improve LLM accuracy without fine-tuning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re


class CSVSchemaAnalyzer:
    """Analyzes CSV structure and generates intelligent context for LLM prompts"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.schema = self._analyze_schema()
    
    def _analyze_schema(self) -> Dict[str, Any]:
        """Comprehensive schema analysis"""
        schema = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "columns": {},
            "numeric_columns": [],
            "text_columns": [],
            "date_columns": [],
            "categorical_columns": [],
            "column_groups": {},
            "sample_values": {}
        }
        
        # Analyze each column
        for col in self.df.columns:
            col_info = self._analyze_column(col)
            schema["columns"][col] = col_info
            
            # Categorize columns
            if col_info["type"] == "numeric":
                schema["numeric_columns"].append(col)
            elif col_info["type"] == "text":
                schema["text_columns"].append(col)
            elif col_info["type"] == "datetime":
                schema["date_columns"].append(col)
            
            if col_info["is_categorical"]:
                schema["categorical_columns"].append(col)
            
            # Store sample values
            schema["sample_values"][col] = col_info["samples"]
        
        # Detect column groups (e.g., First Name + Last Name = name group)
        schema["column_groups"] = self._detect_column_groups()
        
        return schema
    
    def _analyze_column(self, col: str) -> Dict[str, Any]:
        """Analyze individual column"""
        series = self.df[col]
        
        info = {
            "name": col,
            "type": self._detect_type(series),
            "null_count": series.isnull().sum(),
            "null_percentage": (series.isnull().sum() / len(series)) * 100,
            "unique_count": series.nunique(),
            "is_categorical": False,
            "samples": [],
            "stats": {}
        }
        
        # Get sample values (non-null)
        non_null = series.dropna()
        if len(non_null) > 0:
            info["samples"] = non_null.head(5).tolist()
        
        # Determine if categorical
        if info["unique_count"] < 50 and info["unique_count"] < len(series) * 0.5:
            info["is_categorical"] = True
            info["categories"] = series.value_counts().head(10).to_dict()
        
        # Type-specific stats
        if info["type"] == "numeric":
            # Only compute numeric stats if the series is actually numeric
            # This prevents TypeError with mixed-type columns
            try:
                # Try to convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(series, errors='coerce')
                if not numeric_series.dropna().empty:
                    info["stats"] = {
                        "min": float(numeric_series.min()) if not numeric_series.empty else None,
                        "max": float(numeric_series.max()) if not numeric_series.empty else None,
                        "mean": float(numeric_series.mean()) if not numeric_series.empty else None,
                        "median": float(numeric_series.median()) if not numeric_series.empty else None
                    }
                else:
                    info["stats"] = {}
            except Exception:
                info["stats"] = {}
        elif info["type"] == "text":
            try:
                info["stats"] = {
                    "avg_length": series.astype(str).str.len().mean() if not series.empty else 0,
                    "max_length": series.astype(str).str.len().max() if not series.empty else 0
                }
            except Exception:
                info["stats"] = {}
        
        return info
    
    def _detect_type(self, series: pd.Series) -> str:
        """Detect column data type"""
        # Check pandas dtype first
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        
        # Try to infer from content
        non_null = series.dropna()
        if len(non_null) == 0:
            return "text"
        
        # Check if looks like dates
        sample = str(non_null.iloc[0])
        if re.match(r'\d{4}-\d{2}-\d{2}', sample) or re.match(r'\d{2}/\d{2}/\d{4}', sample):
            return "datetime"
        
        # Check if numeric-like
        try:
            pd.to_numeric(non_null.head(10))
            return "numeric"
        except:
            pass
        
        return "text"
    
    def _detect_column_groups(self) -> Dict[str, List[str]]:
        """Detect related columns (e.g., First Name + Last Name)"""
        groups = {}
        
        # Common patterns
        patterns = {
            "name": ["first name", "last name", "full name", "name", "customer name"],
            "address": ["street", "address", "city", "state", "zip", "postal", "country"],
            "contact": ["email", "phone", "mobile", "telephone", "fax"],
            "date": ["date", "created", "updated", "modified", "timestamp"],
            "amount": ["price", "cost", "amount", "total", "subtotal", "tax"]
        }
        
        for group_name, keywords in patterns.items():
            matching_cols = []
            for col in self.df.columns:
                col_lower = col.lower()
                if any(kw in col_lower for kw in keywords):
                    matching_cols.append(col)
            
            if matching_cols:
                groups[group_name] = matching_cols
        
        return groups
    
    def get_prompt_context(self, max_columns: int = 20) -> str:
        """Generate rich context for LLM prompts"""
        context = f"""
📊 DATASET SCHEMA ANALYSIS:

**Overview:**
- Total Rows: {self.schema['total_rows']:,}
- Total Columns: {self.schema['total_columns']}

**Column Details:**
"""
        
        # Show detailed column info
        for i, (col, info) in enumerate(list(self.schema['columns'].items())[:max_columns]):
            context += f"\n{i+1}. **{col}** ({info['type']})"
            
            if info['samples']:
                samples_str = ", ".join([f"'{s}'" for s in info['samples'][:3]])
                context += f"\n   Examples: {samples_str}"
            
            if info['type'] == 'numeric' and info['stats']:
                context += f"\n   Range: {info['stats'].get('min', 'N/A')} to {info['stats'].get('max', 'N/A')}"
            
            if info['is_categorical'] and 'categories' in info:
                top_cats = list(info['categories'].keys())[:3]
                context += f"\n   Categories: {', '.join(map(str, top_cats))}"
        
        # Add column groups
        if self.schema['column_groups']:
            context += "\n\n**Column Groups (Related Fields):**\n"
            for group, cols in self.schema['column_groups'].items():
                context += f"- {group.title()}: {', '.join(cols)}\n"
        
        return context
    
    def suggest_columns_for_query(self, query: str) -> List[str]:
        """Intelligently suggest columns based on user query"""
        query_lower = query.lower()
        suggested = []
        
        # Keyword matching with fuzzy logic
        keywords = {
            'name': ['name', 'customer', 'user', 'person', 'defendant', 'client'],
            'email': ['email', 'mail', 'e-mail'],
            'phone': ['phone', 'mobile', 'telephone', 'contact'],
            'address': ['address', 'street', 'location'],
            'state': ['state', 'province', 'region'],
            'city': ['city', 'town'],
            'date': ['date', 'time', 'created', 'updated'],
            'amount': ['price', 'cost', 'amount', 'total', 'payment']
        }
        
        # Check query for keywords
        for category, terms in keywords.items():
            if any(term in query_lower for term in terms):
                # Find matching columns
                if category in self.schema['column_groups']:
                    suggested.extend(self.schema['column_groups'][category])
        
        # Fuzzy match column names directly
        words = re.findall(r'\b\w{3,}\b', query_lower)
        for word in words:
            for col in self.df.columns:
                if word in col.lower() and col not in suggested:
                    suggested.append(col)
        
        return list(dict.fromkeys(suggested))  # Remove duplicates
    
    def generate_few_shot_examples(self, task_type: str = "query") -> str:
        """Generate relevant few-shot examples based on actual data"""
        examples = ""
        
        if task_type == "query":
            # Generate query examples using actual columns
            if self.schema['column_groups'].get('name'):
                name_cols = self.schema['column_groups']['name']
                examples += f"\nExample: \"show me 10 customers\"\nCode: result = df[{name_cols[:2]}].head(10)\n"
            
            if self.schema['categorical_columns']:
                cat_col = self.schema['categorical_columns'][0]
                sample_val = self.schema['sample_values'][cat_col][0] if self.schema['sample_values'][cat_col] else "value"
                examples += f"\nExample: \"filter by {cat_col}\"\nCode: result = df[df['{cat_col}'] == '{sample_val}']\n"
        
        elif task_type == "analysis":
            if self.schema['numeric_columns']:
                num_col = self.schema['numeric_columns'][0]
                examples += f"\nExample: \"average {num_col}\"\nCode: result = df['{num_col}'].mean()\n"
        
        return examples


def create_enhanced_prompt(df: pd.DataFrame, user_query: str, task_type: str = "query") -> str:
    """Create an enhanced prompt with full schema context"""
    analyzer = CSVSchemaAnalyzer(df)
    
    prompt = f"""You are an expert data analyst. Use the schema below to generate PERFECT pandas code.

USER QUERY: "{user_query}"

{analyzer.get_prompt_context()}

**Suggested Columns for this Query:**
{', '.join(analyzer.suggest_columns_for_query(user_query)) or 'Use your best judgment'}

**Few-Shot Examples:**
{analyzer.generate_few_shot_examples(task_type)}

**CRITICAL RULES:**
1. Use EXACT column names from the schema above
2. Handle missing values with .dropna() or .fillna()
3. Use .str.contains() for text searches with case=False, na=False
4. Store final result in variable 'result'
5. Return a DataFrame whenever possible

Generate ONLY the Python code (no explanations):
"""
    
    return prompt
