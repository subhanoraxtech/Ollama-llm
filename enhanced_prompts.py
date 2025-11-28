"""
Enhanced Agent Wrappers with CSV Intelligence
Simple drop-in replacements that add schema-aware prompting
"""

from csv_intelligence import CSVSchemaAnalyzer
import pandas as pd


def enhance_query_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for data query operations"""
    analyzer = CSVSchemaAnalyzer(df)
    suggested_cols = analyzer.suggest_columns_for_query(question)
    
    prompt = f"""You are a data analysis expert. Follow this ReAct (Reasoning + Acting) process:

1. **Thought**: Analyze the user's request and the dataset schema. Decide which columns to use and what operations to perform.
2. **Action**: Generate the Python code to solve the problem.

USER QUERY: "{question}"

{analyzer.get_prompt_context(max_columns=15)}

**Suggested Columns:** {', '.join(suggested_cols) if suggested_cols else 'Choose appropriate columns'}

**RULES:**
1. Use EXACT column names from schema (case-sensitive!)
2. **SELECT ONLY** the columns explicitly requested by the user. Do not include extra columns.
3. Store result in variable 'result'
4. Handle missing values with .dropna() or .fillna()
5. For text search: use .str.contains('term', case=False, na=False)
6. Output the code inside a ```python block.

**Examples:**
Thought: User wants to see customers. I should select name and email columns.
Action:
```python
result = df[['Name', 'Email']].head(5)
```

Generate response for: "{question}"
"""
    
    return prompt


def enhance_export_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for CSV export operations"""
    analyzer = CSVSchemaAnalyzer(df)
    suggested_cols = analyzer.suggest_columns_for_query(question)
    
    prompt = f"""You are a CSV export specialist. Follow this ReAct process:

1. **Thought**: Analyze what columns and rows the user wants to export. Check exact column names.
2. **Action**: Generate the Python code to create the export dataframe.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**Suggested Columns:** {', '.join(suggested_cols) if suggested_cols else 'Analyze and select columns'}

**RULES:**
1. Use EXACT column names (case-sensitive!)
2. If user specifies columns → SELECT ONLY THOSE
3. If user specifies number → use .head(N)
4. Store result in variable 'result'
5. Output the code inside a ```python block.

**Examples:**
Thought: User wants 100 rows with name and email. I will select 'Name' and 'Email' columns and limit to 100.
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
    
    prompt = f"""You are a data analyst. Follow this ReAct process:

1. **Thought**: Identify the calculation needed (sum, mean, count, etc.) and the correct numeric columns.
2. **Action**: Generate the Python code to perform the analysis.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=15)}

**Numeric Columns:** {', '.join(numeric_cols) if numeric_cols else 'None'}

**RULES:**
1. Use EXACT column names (case-sensitive!)
2. Store result in variable 'result'
3. Use .mean(), .sum(), .count(), .max(), .min()
4. Output the code inside a ```python block.

**Examples:**
Thought: User wants average age. I will use the 'Age' column and calculate the mean.
Action:
```python
result = df['Age'].mean()
```

Generate response for: "{question}"
"""
    
    return prompt
