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
    """Generate enhanced prompt for data query operations with Text-to-SQL capabilities"""
    analyzer = CSVSchemaAnalyzer(df)
    suggested_cols = analyzer.suggest_columns_for_query(question)
    hint_text = get_domain_mappings(df, question)
    
    # Smart "All Columns" Logic
    show_suggestions = True
    q_lower = question.lower()
    if any(w in q_lower for w in ['all', 'every', 'full', 'everything', 'entire']):
        show_suggestions = False
        
    suggestions_text = ', '.join(suggested_cols) if suggested_cols and show_suggestions else 'ALL columns (user wants everything)'
    
    prompt = f"""You are a data analyst converting natural language to pandas code.

USER QUERY: "{question}"

DATASET SCHEMA:
{analyzer.get_prompt_context(max_columns=25)}

**Suggested Columns:** {suggestions_text}

**Column Hints:**
{hint_text}

**CRITICAL RULES - FOLLOW EXACTLY:**

1. **For text/string filtering, ALWAYS use this pattern:**
   ```python
   df[df['ColumnName'].astype(str).str.strip().str.contains('value', case=False, na=False)]
   ```
   
2. **State filtering - include BOTH abbreviation and full name:**
   - Texas: `'TX|Texas'`
   - California: `'CA|California'`
   - New York: `'NY|New York'`

3. **NEVER use `==` for text columns** - it's too strict and will miss matches

4. **Always store result in `result` variable**

5. **Validate your code:**
   - Does it answer the user's question?
   - Are you using the correct columns?
   - Will it return the data the user asked for?

**CORRECT EXAMPLES:**

Example 1: "Show customers from Texas"
```python
# Correct - handles TX, Texas, tx, " TX ", etc.
result = df[df['State'].astype(str).str.strip().str.contains('TX|Texas', case=False, na=False)]
```

Example 2: "Get all data" or "Show everything"
```python
# Correct - return all rows and columns
result = df
```

Example 3: "Customers with balance over 1000"
```python
# Correct - numeric comparison
result = df[df['Balance'] > 1000]
```

Example 4: "Texas customers with balance over 1000"
```python
# Correct - combine filters
mask_state = df['State'].astype(str).str.strip().str.contains('TX|Texas', case=False, na=False)
mask_balance = df['Balance'] > 1000
result = df[mask_state & mask_balance]
```

**WRONG EXAMPLES - DO NOT DO THIS:**
❌ `df[df['State'] == 'TX']` - Too strict!
❌ `df[df['State'].str.contains('TX')]` - Missing astype, strip, case, na handling
❌ Using wrong column names not in the schema

**Your Response Format:**
Thought: [Brief explanation of what you're doing and why]
Action:
```python
# Your code here with comments
result = ...
```

**BEFORE YOU RESPOND:**
- Double-check you're using the correct column names from the schema
- Verify your code will return what the user asked for
- Make sure you're using the MANDATORY pattern for text filtering

Now write the code:
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


def enhance_transformation_prompt(df: pd.DataFrame, question: str, dataframes: list[pd.DataFrame] = None) -> str:
    """Generate enhanced prompt for data transformation operations"""
    
    # Handle Multi-CSV Context
    if dataframes and len(dataframes) > 1:
        context_str = ""
        for i, d in enumerate(dataframes):
            analyzer = CSVSchemaAnalyzer(d)
            context_str += f"\n\n--- DataFrame {i} (variable: `dataframes[{i}]`) ---\n"
            context_str += analyzer.get_prompt_context(max_columns=15)
            
        prompt = f"""You are a data transformation expert. You have access to {len(dataframes)} datasets.

USER REQUEST: "{question}"

DATASETS AVAILABLE:
{context_str}

**Smart Column Hints:**
{get_domain_mappings(dataframes[-1], question)}

**RULES:**
1. **Access**: Use `dataframes[0]`, `dataframes[1]`, etc. to access datasets.
2. **Merging**: `pd.merge(dataframes[0], dataframes[1], on='ID')` or `pd.concat([dataframes[0], dataframes[1]])`.
3. **Result**: Store the final transformed dataframe in `result`.

**Response Format:**
Thought: [Reasoning]
Action:
```python
# Merge example
result = pd.merge(dataframes[0], dataframes[1], on='ID', how='inner')
```
"""
        return prompt

    # Single CSV Context (Fallback)
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
