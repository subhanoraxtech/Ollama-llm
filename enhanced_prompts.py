"""
Enhanced Agent Wrappers with CSV Intelligence
Simple drop-in replacements that add schema-aware prompting
"""

import importlib
from csv_intelligence import CSVSchemaAnalyzer
import pandas as pd

def get_domain_mappings(df: pd.DataFrame, question: str) -> str:
    """
    Shared helper to generate column mapping hints for ALL agents.
    This ensures every agent understands domain terms like 'employer', 'bond', 'payment'.
    """
    q_lower = question.lower()
    column_hints = []
    
    # COMPREHENSIVE Domain Dictionary
    term_mappings = {
        'payment': ['payment', 'pay', 'due', 'amount', 'balance', 'fine', 'cost', 'owe', 'outstanding', 'bill', 'ar', 'dues', 'fee'],
        'name': ['name', 'first', 'last', 'customer', 'defendant', 'person', 'ppl', 'people', 'client'],
        'email': ['email', 'mail', 'e-mail', 'contact'],
        'phone': ['phone', 'mobile', 'cell', 'contact', 'call'],
        'date': ['date', 'time', 'when', 'due', 'deadline', 'year', 'registered', 'joined', 'filed', 'print', 'appearance', 'birth', 'dob'],
        'location': ['state', 'city', 'address', 'location', 'zip', 'county', 'residence'],
        'status': ['status', 'active', 'suspended', 'inactive', 'standing'],
        'employer': ['employer', 'work', 'job', 'company', 'workplace', 'occupation'],
        'bond': ['bond', 'surety', 'insurance', 'premium', 'insurer', 'coverage', 'policy', 'underwriter'],
        'financial': ['billed', 'received', 'invoice', 'paid', 'total', 'revenue', 'collection'],
        'id': ['id', 'identifier', 'number', 'case', 'account', 'reference']
    }
    
    # 1. Term-to-Column Mapping
    for term, keywords in term_mappings.items():
        if any(kw in q_lower for kw in keywords):
            matching_cols = [col for col in df.columns if any(kw in col.lower() for kw in keywords)]
            if matching_cols:
                # Prioritize exact matches
                exact = [c for c in matching_cols if term in c.lower()]
                others = [c for c in matching_cols if term not in c.lower()]
                best_matches = (exact + others)[:3]
                column_hints.append(f"For '{term}' concept: use {', '.join(best_matches)}")
    
    # 2. State/Location Intelligence
    state_map = {
        'tx': 'Texas', 'ca': 'California', 'ny': 'New York', 'fl': 'Florida', 'az': 'Arizona', 
        'nv': 'Nevada', 'ga': 'Georgia', 'co': 'Colorado', 'la': 'Louisiana', 'wa': 'Washington',
        'or': 'Oregon', 'il': 'Illinois', 'nj': 'New Jersey', 'ut': 'Utah', 'ks': 'Kansas',
        'nc': 'North Carolina', 'sc': 'South Carolina', 'va': 'Virginia', 'oh': 'Ohio', 'mi': 'Michigan'
    }
    
    for abbr, full in state_map.items():
        if f" {abbr} " in f" {q_lower} " or f" {full.lower()} " in f" {q_lower} ":
             column_hints.append(f"Hint: User is asking for {full} ({abbr.upper()}). Check State/Location columns.")

    # 3. Contextual Disambiguation
    if 'employer' in q_lower and any(x in q_lower for x in ['in', 'from', 'located']):
        column_hints.append("Hint: User is asking for EMPLOYER location. Use 'EmployerState', 'WorkState', or similar, NOT the defendant's state.")
    
    return "\n".join(column_hints) if column_hints else "Use exact column names from schema"


def enhance_query_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for data query operations"""
    analyzer = CSVSchemaAnalyzer(df)
    suggested_cols = analyzer.suggest_columns_for_query(question)
    hint_text = get_domain_mappings(df, question)
    
    # Smart "All Columns" Logic
    show_suggestions = True
    q_lower = question.lower()
    if any(w in q_lower for w in ['all', 'every', 'full', 'everything', 'entire']):
        show_suggestions = False
        
    suggestions_text = ', '.join(suggested_cols) if suggested_cols and show_suggestions else 'None (Select ALL columns)'
    
    prompt = f"""You are a high-performance data analysis assistant.

USER QUERY: "{question}"

{analyzer.get_prompt_context(max_columns=25)}

**Suggested Columns:** {suggestions_text}

**Smart Column Hints:**
{hint_text}

**INTERPRETATION RULES (Handle Vague Queries):**
1. **"Payment Due" / "Owe"**: Check `Balance > 0` OR `Status == 'Unpaid'`.
2. **"From [State]"**: Filter by State column (handle abbreviations like TX->Texas).
3. **"In [Year]"**: Use string matching `.astype(str).str.contains('2025')` for safety.
4. **"Fix/Clean"**: If user asks to fix/clean in a query, just SHOW the bad data (don't delete).
5. **Typos**: Handle "defandant", "texes", "balence" by mapping to correct columns.

**PERFORMANCE RULES:**
1. **Vectorization**: Use vectorized pandas operations. NEVER use loops.
2. **Select ALL**: Return full dataframe unless specific columns requested.
3. **Result**: Store final result in `result`.

**Response Format:**
Thought: [Reasoning]
Action:
```python
result = ...
```
"""
    return prompt


def enhance_cleaning_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for data cleaning operations"""
    analyzer = CSVSchemaAnalyzer(df)
    hint_text = get_domain_mappings(df, question)
    
    prompt = f"""You are a data cleaning expert. Clean the dataset intelligently.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**Smart Column Hints:**
{hint_text}

**AUTO-CLEANING RULES (If request is vague):**
1. **"Clean data"**: Remove duplicates (`drop_duplicates`), trim whitespace (`str.strip`), and fill NaNs in critical columns.
2. **"Fix emails"**: Lowercase and remove invalid formats.
3. **"Standardize"**: Title case names (`str.title()`), upper case states (`str.upper()`).

**EXECUTION RULES:**
1. **Efficiency**: Use vectorized string operations.
2. **Safety**: Create a copy `df_clean = df.copy()`.
3. **Result**: Store the cleaned dataframe in `result`.

**Response Format:**
Thought: [Reasoning]
Action:
```python
df_clean = df.copy()
# ... cleaning steps ...
result = df_clean
```
"""
    return prompt


def enhance_transformation_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for data transformation operations"""
    analyzer = CSVSchemaAnalyzer(df)
    hint_text = get_domain_mappings(df, question)
    
    prompt = f"""You are a data transformation expert. Reshape or modify the dataset.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**Smart Column Hints:**
{hint_text}

**INTERPRETATION RULES:**
1. **"Group by [X]"**: Use `groupby(X)`.
2. **"Summarize [X] by [Y]"**: `df.groupby(Y)[X].sum()` (or count/mean).
3. **"Filter [Condition]"**: Apply boolean indexing.
4. **"Pivot"**: Use `pivot_table`.

**Response Format:**
Thought: [Reasoning]
Action:
```python
result = df.groupby('State')['Balance'].sum().reset_index()
```
"""
    return prompt


def enhance_analysis_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for analysis operations"""
    analyzer = CSVSchemaAnalyzer(df)
    numeric_cols = analyzer.schema['numeric_columns']
    hint_text = get_domain_mappings(df, question)
    
    prompt = f"""You are a senior data analyst. Provide accurate insights.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=15)}

**Numeric Columns:** {', '.join(numeric_cols)}

**Smart Column Hints:**
{hint_text}

**AUTO-ANALYSIS RULES (If request is vague):**
1. **"Analyze"**: Provide `df.describe()` and `df.info()`.
2. **"Trends"**: Group by Date/Year and sum/count.
3. **"Distribution"**: Use `value_counts()` for categorical, `describe()` for numeric.

**Response Format:**
Thought: [Reasoning]
Action:
```python
result = df['Balance'].describe()
```
"""
    return prompt


def enhance_advanced_analytics_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for advanced analytics"""
    analyzer = CSVSchemaAnalyzer(df)
    numeric_cols = analyzer.schema['numeric_columns']
    hint_text = get_domain_mappings(df, question)
    
    prompt = f"""You are a data scientist. Perform advanced segmentation or pattern extraction.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**Smart Column Hints:**
{hint_text}

**INTERPRETATION RULES:**
1. **"Segment/Cluster"**: Use `KMeans` (if numeric) or `qcut` (if simple).
2. **"Outliers"**: Find rows with Z-score > 3 or outside IQR.
3. **"High Value"**: Top 10% by Balance/Bill (`quantile(0.9)`).

**Response Format:**
Thought: [Reasoning]
Action:
```python
result = df.describe()
```
"""
    return prompt


def enhance_export_prompt(df: pd.DataFrame, question: str) -> str:
    """Generate enhanced prompt for CSV export"""
    analyzer = CSVSchemaAnalyzer(df)
    suggested_cols = analyzer.suggest_columns_for_query(question)
    hint_text = get_domain_mappings(df, question)
    
    prompt = f"""You are a CSV export specialist. Prepare data for export.

USER REQUEST: "{question}"

{analyzer.get_prompt_context(max_columns=20)}

**Smart Column Hints:**
{hint_text}

**RULES:**
1. **Exact Columns**: Use EXACT column names.
2. **User Intent**: If user specifies columns, SELECT ONLY THOSE. If "all", select all.
3. **Row Limits**: If "top 100", use `.head(100)`.
4. **Result**: Store dataframe in `result`.

**Response Format:**
Thought: [Reasoning]
Action:
```python
result = df.head(100)
```
"""
    return prompt
