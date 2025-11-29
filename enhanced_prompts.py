"""
Enhanced Agent Wrappers with CSV Intelligence
Simple drop-in replacements that add schema-aware prompting
"""

import importlib
from csv_intelligence import CSVSchemaAnalyzer
import pandas as pd


def enhance_query_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for data query operations with query preprocessing"""
    analyzer = CSVSchemaAnalyzer(df)
    suggested_cols = analyzer.suggest_columns_for_query(question)
    
    # Preprocess query to map user terms to column names
    q_lower = question.lower()
    column_hints = []
    
    # Map common user terms to actual columns (Expanded for Defendants/Legal Domain)
    term_mappings = {
        'payment': ['payment', 'pay', 'due', 'amount', 'balance', 'fine', 'cost', 'owe', 'outstanding', 'bill'],
        'name': ['name', 'first', 'last', 'customer', 'defendant', 'person', 'ppl', 'people'],
        'email': ['email', 'mail', 'e-mail'],
        'date': ['date', 'time', 'when', 'due', 'deadline', 'year', 'registered', 'joined', 'filed'],
        'location': ['state', 'city', 'address', 'location', 'zip', 'county'],
        'status': ['status', 'active', 'suspended', 'inactive']
    }
    
    for term, keywords in term_mappings.items():
        if any(kw in q_lower for kw in keywords):
            matching_cols = [col for col in df.columns if any(kw in col.lower() for kw in keywords)]
            if matching_cols:
                column_hints.append(f"For '{term}': use {', '.join(matching_cols[:3])}")
    
    # Specific logic for State/Location abbreviations
    state_map = {'tx': 'Texas', 'ca': 'California', 'ny': 'New York', 'fl': 'Florida', 'az': 'Arizona'}
    for abbr, full in state_map.items():
        if f" {abbr} " in f" {q_lower} " or f" {full.lower()} " in f" {q_lower} ":
             column_hints.append(f"Hint: User is asking for {full} ({abbr.upper()}). Check State/Location columns.")

    hint_text = "\n".join(column_hints) if column_hints else "Use exact column names from schema"
    
    # Determine if we should show suggestions
    show_suggestions = True
    if any(w in q_lower for w in ['all', 'every', 'full', 'everything']):
        show_suggestions = False
        
    suggestions_text = ', '.join(suggested_cols) if suggested_cols and show_suggestions else 'None (Select ALL columns)'
    
    prompt = f"""You are a high-performance data analysis assistant specialized in legal and financial data. 

USER QUERY: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**Suggested Columns:** {suggestions_text}

**Column Mapping Hints:**
{hint_text}

**PERFORMANCE & ACCURACY RULES:**
1. **Vectorization**: Use vectorized pandas operations. NEVER use loops.
2. **Case Sensitivity**: Column names are Case-Sensitive.
3. **Payment Due Logic**: If user asks for "payment due", "outstanding", "owe", or "unpaid":
   - Check if a 'Balance' or 'Amount Due' column is > 0.
   - OR check if 'Status' is 'Unpaid'/'Pending'.
4. **Dates/Years**: 
   - For "in 2025": Convert column to datetime or use string matching `.astype(str).str.contains('2025', na=False)`.
5. **Fuzzy Matching**: Handle typos (e.g. "texes" -> "Texas", "defandants" -> "Defendant").
6. **Select ALL Columns**: Return full dataframe unless specific columns requested.
7. **Result**: Store final result in `result`.

**Response Format:**
Thought: [Reasoning]
Action:
```python
# Efficient pandas code
result = ...
```

**Few-Shot Examples:**

Thought: User wants defendants from TX with payment due. I'll filter by State 'TX' and Balance > 0.
Action:
```python
# Assuming columns 'State' and 'Balance' exist
result = df[(df['State'] == 'TX') & (df['Balance'] > 0)]
```

Thought: User wants defendants registered in 2025.
Action:
```python
# String match is safer for mixed formats
result = df[df['RegistrationDate'].astype(str).str.contains('2025', na=False)]
```

Thought: Top 10 highest unpaid defendants.
Action:
```python
result = df[df['Balance'] > 0].nlargest(10, 'Balance')
```

Generate response for: "{question}"
"""
    
    return prompt


def enhance_export_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for CSV export operations"""
    analyzer = CSVSchemaAnalyzer(df)
    suggested_cols = analyzer.suggest_columns_for_query(question)
    
    prompt = f"""You are a CSV export specialist. Your goal is to prepare data for export precisely as requested.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**Suggested Columns:** {', '.join(suggested_cols) if suggested_cols else 'Analyze and select columns'}

**RULES:**
1. **Exact Columns**: Use EXACT column names (case-sensitive!).
2. **User Intent**: If user specifies columns, SELECT ONLY THOSE. If they want "all", select all.
3. **Row Limits**: If user specifies a number (e.g. "top 100"), use `.head(N)`.
4. **Result**: Store the final dataframe in variable `result`.
5. **Efficiency**: Do not print the dataframe, just assign it to `result`.

**Response Format:**
Thought: [Reasoning]
Action:
```python
result = ...
```

**Examples:**
Thought: User wants 100 rows with name and email.
Action:
```python
result = df[['Name', 'Email']].head(100)
```

Generate response for: "{question}"
"""
    
    return prompt


def enhance_analysis_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for analysis operations"""
    analyzer = CSVSchemaAnalyzer(df)
    numeric_cols = analyzer.schema['numeric_columns']
    
    prompt = f"""You are a senior data analyst. Perform the requested calculation accurately.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=15)}

**Numeric Columns:** {', '.join(numeric_cols) if numeric_cols else 'None'}

**RULES:**
1. **Calculation**: Identify the correct aggregation (sum, mean, count, etc.).
2. **Columns**: Use EXACT column names.
3. **Grouping**: If the user asks to "break down by" or "per", use `groupby()`.
4. **Result**: Store the final result (number, series, or dataframe) in variable `result`.
5. **Efficiency**: Use built-in pandas methods (`.mean()`, `.sum()`) rather than manual calculation.

**Response Format:**
Thought: [Reasoning]
Action:
```python
result = ...
```

**Examples:**
Thought: User wants average age.
Action:
```python
result = df['Age'].mean()
```

Generate response for: "{question}"
"""
    
    return prompt


def enhance_cleaning_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for data cleaning operations"""
    analyzer = CSVSchemaAnalyzer(df)
    
    prompt = f"""You are a data cleaning expert. Your task is to clean/fix the dataset using pandas.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**RULES:**
1. **Efficiency**: Use vectorized string operations (`.str.strip()`, `.str.title()`, `.str.replace()`).
2. **Filtering**: If the user specifies a subset (e.g. "from 2025"), FILTER the dataframe first or apply changes only to those rows.
3. **Safety**: Create a copy if necessary.
4. **Result**: Store the cleaned dataframe (or the specific affected rows) in `result`.
5. **Missing Values**: If asked to handle missing values, use `.dropna()` or `.fillna()`.

**Response Format:**
Thought: [Reasoning]
Action:
```python
# Cleaning code
df_clean = df.copy()
df_clean['Name'] = df_clean['Name'].str.title()
result = df_clean
```

Generate response for: "{question}"
"""
    return prompt


def enhance_transformation_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for data transformation operations"""
    analyzer = CSVSchemaAnalyzer(df)
    
    prompt = f"""You are a data transformation expert. Your task is to reshape, filter, or modify the dataset.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**RULES:**
1. **Operations**: Handle merging, filtering, sorting, pivoting, melting, or adding columns.
2. **Complex Logic**: If the user asks for "Group by X and list top Y", use `groupby()` and `apply()`.
3. **Result**: Store the transformed dataframe in `result`.
4. **Efficiency**: Use optimized pandas methods.

**Response Format:**
Thought: [Reasoning]
Action:
```python
# Transformation code
result = df.groupby('City')['Spend'].sum().reset_index()
```

Generate response for: "{question}"
"""
    return prompt


def enhance_advanced_analytics_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for advanced analytics (segmentation, outliers, etc.)"""
    analyzer = CSVSchemaAnalyzer(df)
    numeric_cols = analyzer.schema['numeric_columns']
    
    prompt = f"""You are a data scientist. Your task is to perform advanced analytics, segmentation, or pattern extraction.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**Numeric Columns:** {', '.join(numeric_cols)}

**RULES:**
1. **Segmentation**: If asked to "segment" or "cluster", use `sklearn.cluster.KMeans` or simple quantile-based segmentation (`pd.qcut`).
2. **Outliers**: Use IQR or Z-score logic if asked for outliers.
3. **Trends**: Use aggregation and sorting to find trends.
4. **Result**: Store the analysis result (dataframe or summary) in `result`.
5. **Visualization**: Do NOT generate charts, just the data.

**Response Format:**
Thought: [Reasoning]
Action:
```python
# Analytics code
result = df.describe()
```

Generate response for: "{question}"
"""
    return prompt
