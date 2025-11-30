import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from docx import Document
import tempfile
import json
import re
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import datetime
import shutil
import uuid

# Import Data Analysis Agents
from agents.cleaning_agent import CleaningAgent
from agents.transformation_agent import TransformationAgent
from agents.analysis_agent import AnalysisAgent as DataAnalysisAgent
from agents.visualization_agent import VisualizationAgent
from agents.export_agent import ExportAgent
from agents.advanced_analytics_agent import AdvancedAnalyticsAgent

# Import CSV Intelligence System
import importlib
import enhanced_prompts
importlib.reload(enhanced_prompts)

# Bind functions from the reloaded module
enhance_query_prompt = enhanced_prompts.enhance_query_prompt
enhance_export_prompt = enhanced_prompts.enhance_export_prompt
enhance_analysis_prompt = enhanced_prompts.enhance_analysis_prompt
enhance_cleaning_prompt = enhanced_prompts.enhance_cleaning_prompt
enhance_transformation_prompt = enhanced_prompts.enhance_transformation_prompt
enhance_advanced_analytics_prompt = enhanced_prompts.enhance_advanced_analytics_prompt
from csv_intelligence import CSVSchemaAnalyzer
from csv_comparator import CSVComparator

# Import Query Rephraser for RAFT
from query_rephraser import QueryRephraser

# === Models ===
llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.7,
    num_gpu=999,
    num_thread=8,
    num_ctx=8192,
)

code_llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    num_gpu=999,
    num_thread=8,
    num_ctx=8192,
)

embed_model = OllamaEmbeddings(model="mxbai-embed-large")

# Initialize Query Rephraser for RAFT
query_rephraser = QueryRephraser(llm)

# Initialize Data Analysis Agents
cleaning_agent = CleaningAgent()
transformation_agent = TransformationAgent()
data_analysis_agent = DataAnalysisAgent()
visualization_agent = VisualizationAgent()
export_agent = ExportAgent()
advanced_analytics_agent = AdvancedAnalyticsAgent()

# === State ===
class State(TypedDict):
    messages: Annotated[list, add_messages]
    dataframes: List[pd.DataFrame]
    agent_decision: Optional[str]
    csv_files: List[dict]
    filtered_df: Optional[pd.DataFrame]  # Added for data persistence

# === File Parsing ===
def parse_file(file):
    ext = file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    try:
        if ext == 'csv':
            return {"type": "csv", "data": pd.read_csv(tmp_path)}
        elif ext == 'pdf':
            text = "\n".join([p.extract_text() or "" for p in PdfReader(tmp_path).pages])
            return {"type": "text", "data": text}
        elif ext in ['docx', 'doc']:
            text = "\n".join([p.text for p in Document(tmp_path).paragraphs if p.text.strip()])
            return {"type": "text", "data": text}
    finally:
        os.unlink(tmp_path)
    return None

# === RAG Chain with RAFT ===
def create_raft_chain(vectorstore):
    """RAFT-enhanced RAG chain with multi-query retrieval and chain-of-thought"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    def raft_retrieval(question: str) -> str:
        """Multi-query retrieval with relevance filtering"""
        # Generate query variations
        query_variations = query_rephraser.rephrase_for_documents(question)
        
        # Retrieve documents for each variation
        all_docs = []
        seen_content = set()
        
        for query_var in query_variations:
            docs = retriever.get_relevant_documents(query_var)
            for doc in docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        # Filter for relevance
        relevant_docs = []
        for doc in all_docs[:15]:
            q_terms = set(question.lower().split())
            doc_terms = set(doc.page_content.lower().split())
            if len(q_terms & doc_terms) >= 2:
                relevant_docs.append(doc)
        
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs[:5]])
        return context if context else "No relevant information found."
    
    template = """You are a helpful assistant analyzing documents. Follow this process:

**Think**: Analyze the context and identify relevant information
**Answer**: Provide a clear, accurate answer

Context:
{context}

Question: {question}

Response Format:
**Thought**: [Your analysis]
**Answer**: [Your detailed answer]

If the context doesn't contain relevant information, say so clearly.

Your Response:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": raft_retrieval, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

# ==================== AGENT SYSTEM ====================

# Agent 1: Router Agent - Decides which agent to use
def router_agent(state: State) -> dict:
    """Analyzes the user's question and routes to appropriate agent"""
    question = state["messages"][-1].content
    has_csv = len(state.get("dataframes", [])) > 0
    has_multiple_csv = len(state.get("dataframes", [])) > 1
    has_docs = st.session_state.get("vectorstore") is not None
    
    decision_prompt = f"""You are an intelligent routing system. Analyze the user's request carefully and select the best agent.

USER REQUEST: "{question}"

AVAILABLE DATA:
- CSV Datasets: {len(state.get("dataframes", []))} files loaded
- Documents (PDF/DOCX): {"YES" if has_docs else "NO"}

AGENTS AND THEIR SPECIALTIES:

1. **cleaning_agent** - Use when user wants to:
   - Remove duplicates, fix formatting, clean data
   - Handle missing values, trim whitespace
   - Standardize columns, fix emails/phones
   - Keywords: "clean", "remove duplicates", "fix", "standardize", "missing values"

2. **transformation_agent** - Use when user wants to:
   - Merge/join CSV files, filter, sort, rename columns
   - Add/combine/split columns, pivot/melt data
   - Keywords: "merge", "join", "filter", "sort", "rename", "pivot"

3. **analysis_agent** - Use when user wants to:
   - Statistical analysis, grouping, correlation
   - Compare datasets, detect duplicates across files
   - Keywords: "statistics", "group by", "correlation", "compare", "analyze"

4. **visualization_agent** - Use when user wants to:
   - Create charts, graphs, plots, heatmaps
   - Visualize data
   - Keywords: "chart", "graph", "plot", "visualize", "heatmap", "histogram"

5. **advanced_analytics_agent** - Use when user wants to:
   - Customer segmentation, outlier detection
   - Validate emails/phones, churn prediction
   - Keywords: "segment", "cluster", "outliers", "validate", "churn", "high-value"

6. **export_agent** - Use when user wants to:
   - Export to CSV/Excel/JSON, generate reports
   - Keywords: "export", "download", "save", "report", "pdf", "excel"

7. **data_query_agent** - Use when user wants to:
   - View, show, display, or list data
   - Get specific rows or columns
   - Keywords: "show", "display", "give me", "find", "get", "list"

8. **document_agent** - Use when user asks about:
   - Content from uploaded PDF or DOCX files
   - Keywords: "document", "pdf", "what does it say", "summarize"

RULES:
- Clean/fix data → cleaning_agent
- Merge/transform/filter → transformation_agent  
- Statistics/analysis → analysis_agent
- Charts/graphs → visualization_agent
- ML/segmentation → advanced_analytics_agent
- Export/save → export_agent
- Show/display data → data_query_agent
- Documents → document_agent

Respond with ONLY ONE of these agent names:
cleaning_agent
transformation_agent
analysis_agent
visualization_agent
advanced_analytics_agent
export_agent
data_query_agent
document_agent

YOUR DECISION (one word only):"""

    try:
        response = llm.invoke(decision_prompt)
        agent_name = response.content.strip().lower() if hasattr(response, 'content') else "data_query_agent"
        
        # Extract agent name
        valid_agents = ['cleaning_agent', 'transformation_agent', 'analysis_agent', 'visualization_agent', 
                       'advanced_analytics_agent', 'export_agent', 'data_query_agent', 'document_agent']
        
        # Find matching agent
        for agent in valid_agents:
            if agent in agent_name:
                return {"agent_decision": agent}
        
        # Default to data_query_agent if no match
        return {"agent_decision": "data_query_agent"}
        
    except Exception as e:
        # Fallback to data_query_agent on error
        return {"agent_decision": "data_query_agent"}

# Agent 2: Data Query Agent - Retrieves and displays data
def data_query_agent(state: State) -> dict:
    """Handles data retrieval queries with intelligent analysis"""
    question = state["messages"][-1].content
    
    # CRITICAL: Use filtered_df if available (from previous operations), otherwise use original dataframe
    df = state.get("filtered_df")
    using_filtered = df is not None
    
    # Get all available dataframes
    dataframes = state.get("dataframes", [])
    
    # Determine execution mode
    multi_csv_mode = False
    if df is None:
        if not dataframes:
            return {"messages": [AIMessage(content="❌ No dataset loaded. Please upload a CSV file first to query data.")]}
        
        if len(dataframes) > 1:
            multi_csv_mode = True
            # Default to the last one for single-df operations, but we'll expose all
            df = dataframes[-1]
        else:
            df = dataframes[0]
    
    if df is None:
        return {"messages": [AIMessage(content="❌ No dataset loaded. Please upload a CSV file first to query data.")]}
    
    # Get chat history for context
    recent_context = []
    if len(state["messages"]) > 1:
        for msg in state["messages"][-3:-1]:
            if hasattr(msg, 'content'):
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                recent_context.append(f"{role}: {msg.content[:100]}")
    
    context_note = "\n".join(recent_context) if recent_context else ""
    
    # Generate Prompt
    if multi_csv_mode and not using_filtered:
        # MULTI-CSV PROMPT GENERATION
        prompt_context = ""
        for i, d in enumerate(dataframes):
            analyzer = CSVSchemaAnalyzer(d)
            prompt_context += f"\n\n--- DataFrame {i} (variable: dataframes[{i}]) ---\n"
            prompt_context += analyzer.get_prompt_context(max_columns=15)
            
        code_prompt = f"""You are a data analysis expert. You have access to {len(dataframes)} datasets.
        
DATASETS AVAILABLE:
{prompt_context}

USER QUERY: "{question}"

CONTEXT:
{context_note}

INSTRUCTIONS:
1. You have access to a list of dataframes named `dataframes`.
2. `dataframes[0]` is the first file, `dataframes[1]` is the second, etc.
3. Analyze the user's request to decide which dataframe(s) to use.
4. Write Python code to answer the query.
5. Store the final result (dataframe or value) in a variable named `result`.

Response Format:
**Thought**: [Your analysis]
**Action**: [Python code]
```python
# Your code here
result = ...
```
"""
    else:
        # SINGLE CSV / FILTERED DF PROMPT
        enhanced_question = question
        if context_note:
            enhanced_question = f"""CONTEXT: 
Recent conversation:
{context_note}

CURRENT REQUEST: {question}
"""
            if using_filtered:
                enhanced_question += f"\nNote: Use the current filtered dataset (df) which has {len(df)} rows."
        
        # Use enhanced schema-aware prompt
        code_prompt = enhance_query_prompt(df, enhanced_question)

    try:
        response = code_llm.invoke(code_prompt)
        full_response = response.content if hasattr(response, 'content') else str(response)
        
        # Extract Thought
        thought = ""
        if "Thought:" in full_response:
            parts = full_response.split("Thought:")
            if len(parts) > 1:
                thought_part = parts[1]
                if "Action:" in thought_part:
                    thought = thought_part.split("Action:")[0].strip()
                else:
                    thought = thought_part.split("\n")[0].strip()
        
        # Extract Code
        code = full_response
        if "```" in full_response:
            code = full_response.split("```")[1]
            if "python" in code.split("\n")[0]:
                code = "\n".join(code.split("\n")[1:])
            code = code.strip("`").strip()
        
        # Execute
        # CRITICAL: Pass 'dataframes' to namespace so code can access all files
        namespace = {"df": df, "dataframes": dataframes, "pd": pd, "np": np}
        exec(code, {}, namespace)
        result = namespace.get("result")
        
        if isinstance(result, pd.DataFrame) and len(result) > 0:
            # Check if we have the columns user asked for
            requested_cols = result.columns.tolist()
            
            # Generate a nice response showing what we found
            sample_rows = []
            for idx, row in result.head(3).iterrows():
                row_str = " | ".join([f"{col}: {row[col]}" for col in requested_cols[:4]])  # Show first 4 cols
                sample_rows.append(f"• {row_str}")
            
            examples = '\n'.join(sample_rows)
            
            thought_display = f"**🧠 Thought:** _{thought}_\n\n" if thought else ""
            
            answer = f"{thought_display}Found {len(result)} records with {len(requested_cols)} columns ({', '.join(requested_cols)}):\n\n{examples}\n\nShowing all in the table below."
            
            return {"messages": [AIMessage(content=answer)], "filtered_df": result}
        else:
             return {"messages": [AIMessage(content=f"Result: {result}")]}

    except Exception as e:
        # Smart fallback
        return smart_fallback_query(df, question)

def fuzzy_match_columns(df: pd.DataFrame, user_keywords: list) -> list:
    """
    Intelligently match user keywords to actual dataframe columns using fuzzy matching.
    Handles typos, partial matches, and variations.
    
    Examples:
    - "nam" matches "Name", "First Name", "Last Name"
    - "emal" matches "Email", "CustomerEmail"
    - "addres" matches "Address", "Street Address"
    """
    matched_cols = []
    
    def similarity_score(keyword: str, column: str) -> float:
        """Calculate similarity between keyword and column name (0-1 scale)"""
        keyword = keyword.lower()
        column = column.lower()
        
        # Exact match
        if keyword == column:
            return 1.0
        
        # Keyword is substring of column
        if keyword in column:
            return 0.9
        
        # Column is substring of keyword (partial typing)
        if column in keyword:
            return 0.85
        
        # Check if keyword is partial match (e.g., "nam" in "name")
        if len(keyword) >= 3:
            for i in range(len(column) - len(keyword) + 1):
                if column[i:i+len(keyword)] == keyword:
                    return 0.8
        
        # Simple character overlap ratio
        common_chars = sum(1 for c in keyword if c in column)
        overlap_ratio = common_chars / max(len(keyword), len(column))
        
        # Levenshtein-like: count differences
        if len(keyword) > 2 and len(column) > 2:
            max_len = max(len(keyword), len(column))
            differences = abs(len(keyword) - len(column))
            for i in range(min(len(keyword), len(column))):
                if keyword[i] != column[i]:
                    differences += 1
            similarity = 1 - (differences / max_len)
            return max(overlap_ratio, similarity)
        
        return overlap_ratio
    
    # For each keyword, find best matching columns
    for keyword in user_keywords:
        best_matches = []
        for col in df.columns:
            score = similarity_score(keyword, col)
            if score >= 0.6:  # Threshold for fuzzy match
                best_matches.append((col, score))
        
        # Sort by score and add top matches
        best_matches.sort(key=lambda x: x[1], reverse=True)
        for col, score in best_matches:
            if col not in matched_cols:
                matched_cols.append(col)
    
    return matched_cols

def smart_fallback_query(df: pd.DataFrame, question: str) -> dict:
    """Intelligent fallback with FUZZY MATCHING for typos and partial words"""
    q_lower = question.lower()
    
    # Extract number from query
    numbers = re.findall(r'\d+', question)
    n = int(numbers[0]) if numbers else 10
    
    # Extract potential column keywords from user's question
    # Remove common words and extract meaningful terms
    stop_words = {'show', 'give', 'me', 'get', 'list', 'find', 'with', 'their', 'the', 'a', 'an', 'and', 'or', 'of', 'to', 'from', 'in', 'for', 'on'}
    words = re.findall(r'\b\w+\b', q_lower)
    potential_keywords = [w for w in words if w not in stop_words and len(w) >= 3 and not w.isdigit()]
    
    # Enhanced keyword mapping with fuzzy-friendly terms
    column_keywords = {
        'name': ['name', 'nam', 'fullname', 'customer', 'defendant', 'user', 'person'],
        'email': ['email', 'emal', 'mail', 'e-mail', 'emailaddress'],
        'state': ['state', 'stat', 'province', 'region'],
        'address': ['address', 'addres', 'addr', 'street', 'location'],
        'phone': ['phone', 'phon', 'telephone', 'mobile', 'cell', 'contact'],
        'city': ['city', 'town', 'cty'],
        'zip': ['zip', 'postal', 'postcode', 'zipcode'],
        'payment': ['payment', 'pay', 'due', 'amount', 'balance', 'cost', 'price'],
        'date': ['date', 'time', 'when', 'due', 'deadline']
    }
    # Check if user wants ALL columns
    if any(w in q_lower for w in ['all', 'every', 'full', 'everything']):
        # Skip column filtering and return all columns
        result = df.head(n)
        return {
            "messages": [AIMessage(content=f"Here are the first {n} rows with ALL columns:")],
            "filtered_df": result
        }
    
    # Step 1: Try keyword-based matching first
    requested_cols = []
    user_mentioned_fields = []
    
    for field, keywords in column_keywords.items():
        if any(kw in q_lower for kw in keywords):
            user_mentioned_fields.append(field)
            
            # Find matching columns with scoring
            matches = []
            for col in df.columns:
                col_lower = col.lower()
                for kw in keywords:
                    if kw in col_lower:
                        # Score: 1.0 for exact match, 0.8 for starts with, 0.5 for contains
                        score = 0.5
                        if col_lower == kw: score = 1.0
                        elif col_lower.startswith(kw): score = 0.8
                        elif kw in col_lower: score = 0.6
                        
                        matches.append((col, score))
                        break # Count each column only once per field
            
            # Sort by score and take top 2
            matches.sort(key=lambda x: x[1], reverse=True)
            for col, score in matches[:2]:
                if col not in requested_cols:
                    requested_cols.append(col)
    
    # Step 2: If keyword matching didn't work well, try FUZZY MATCHING
    if len(requested_cols) == 0 and len(potential_keywords) > 0:
        # Use fuzzy matching on extracted keywords
        fuzzy_matched = fuzzy_match_columns(df, potential_keywords)
        requested_cols.extend([col for col in fuzzy_matched if col not in requested_cols])
    
    # Step 3: If we STILL have nothing, try matching common field names
    if len(requested_cols) == 0:
        # Look for common columns that users typically want
        common_patterns = ['name', 'email', 'phone', 'address', 'state', 'city', 'id']
        for pattern in common_patterns:
            for col in df.columns:
                if pattern in col.lower() and col not in requested_cols:
                    requested_cols.append(col)
                    if len(requested_cols) >= 5:  # Limit to 5 columns
                        break
            if len(requested_cols) >= 5:
                break
    
    # If we found specific columns, use them
    if requested_cols:
        result = df[requested_cols].head(n)
        
        # Generate a nice response showing what we found
        sample_rows = []
        for idx, row in result.head(3).iterrows():
            row_str = " | ".join([f"{col}: {row[col]}" for col in requested_cols[:4]])  # Show first 4 cols
            sample_rows.append(f"• {row_str}")
        
        examples = '\n'.join(sample_rows)
        
        # Smart response based on what we matched
        if user_mentioned_fields:
            fields_str = ", ".join(user_mentioned_fields)
            answer = f"✅ Found {len(result)} records matching your request for **{fields_str}**!\n\n**Columns:** {', '.join(requested_cols)}\n\n**Sample data:**\n{examples}\n\nShowing all in the table below."
        else:
            answer = f"Found {len(result)} records with {len(requested_cols)} columns ({', '.join(requested_cols)}):\n\n{examples}\n\nShowing all in the table below."
        
        return {"messages": [AIMessage(content=answer)], "filtered_df": result}
    
    # Default: show first N rows with all columns
    result = df.head(n)
    return {
        "messages": [AIMessage(content=f"Here are the first {n} rows from the dataset (columns: {', '.join(df.columns[:5])}):")],
        "filtered_df": result
    }

def csv_export_agent(state: State) -> dict:
    """Handles CSV file creation with STRICT column and row selection"""
    question = state["messages"][-1].content
    df = state["dataframes"][-1] if state["dataframes"] else None
    
    if df is None:
        return {"messages": [AIMessage(content="No dataset loaded. Please upload a CSV file first.")]}
    
    # Use enhanced schema-aware prompt
    export_prompt = enhance_export_prompt(df, question)
    
    try:
        response = code_llm.invoke(export_prompt)
        full_response = response.content if hasattr(response, 'content') else str(response)
        
        # Extract Thought
        thought = ""
        if "Thought:" in full_response:
            parts = full_response.split("Thought:")
            if len(parts) > 1:
                thought_part = parts[1]
                if "Action:" in thought_part:
                    thought = thought_part.split("Action:")[0].strip()
                else:
                    thought = thought_part.split("\n")[0].strip()
        
        # Extract Code
        code = full_response
        if "```" in full_response:
            code = full_response.split("```")[1]
            if "python" in code.split("\n")[0]:
                code = "\n".join(code.split("\n")[1:])
            code = code.strip("`").strip()
        
        print("Generated code:\n", code)  # Debug in terminal
        
        # Execute safely
        local_ns = {"df": df, "pd": pd, "np": np}
        exec(code, {}, local_ns)
        result = local_ns.get("result")
        
        if result is None or not isinstance(result, (pd.DataFrame, pd.Series)):
            raise ValueError("No 'result' DataFrame generated")
        
        if isinstance(result, pd.Series):
            result = result.to_frame()
            
        csv_data = result.to_csv(index=False)
        filename = f"export_{len(result)}_records.csv"
        
        thought_display = f"**🧠 Thought:** _{thought}_\n\n" if thought else ""
        
        answer = f"""{thought_display}**CSV Export Ready!**

**Rows:** {len(result):,}  
**Columns:** {len(result.columns)} → `{', '.join(result.columns)}`

**Your request:** {question}

Download your custom CSV below"""

        return {
            "messages": [AIMessage(content=answer)],
            "filtered_df": result,
            "csv_files": [{"name": filename, "data": csv_data, "rows": len(result)}]
        }
        
    except Exception as e:
        print(f"Export failed: {e}")
        # Smart fallback with column guessing
        cols_to_try = []
        q_lower = question.lower()
        if any(w in q_lower for w in ['email', 'mail']):
            cols_to_try += [c for c in df.columns if 'email' in c.lower()]
        if any(w in q_lower for w in ['name', 'customer', 'user']):
            cols_to_try += [c for c in df.columns if any(x in c.lower() for x in ['name', 'customer', 'user'])]
        if 'state' in q_lower:
            cols_to_try += [c for c in df.columns if 'state' in c.lower()]
        
        cols_to_try = list(dict.fromkeys(cols_to_try))[:5]  # dedupe & limit
        if not cols_to_try:
            cols_to_try = df.columns[:5].tolist()
            
        n = 200
        numbers = re.findall(r'\d+', question)
        if numbers:
            n = min(int(numbers[0]), len(df))
            
        result = df[cols_to_try].head(n) if len(cols_to_try) > 1 else df[cols_to_try].head(n).to_frame()
        csv_data = result.to_csv(index=False)
        
        return {
            "messages": [AIMessage(content=f"Created CSV with {len(result)} rows using best matching columns:\n{', '.join(result.columns)}\n\nDownload below!")],
            "filtered_df": result,
            "csv_files": [{"name": f"custom_export_{len(result)}.csv", "data": csv_data, "rows": len(result)}]
        }

# Agent 4: Analysis Agent - Performs calculations

def find_columns(df, keywords):
    """Find best matching columns by keyword"""
    keywords = [k.lower().strip() for k in keywords.replace(" and ", ",").split(",")]
    matches = []
    for kw in keywords:
        for col in df.columns:
            if kw in col.lower() or col.lower() in kw:
                matches.append(col)
    return list(dict.fromkeys(matches))  # preserve order, remove dupes

def analysis_agent(state: State) -> dict:
    """Performs data analysis with proper result interpretation"""
    question = state["messages"][-1].content
    df = state["dataframes"][-1] if state["dataframes"] else None
    
    if df is None:
        return {"messages": [AIMessage(content="❌ No dataset loaded. Please upload a CSV file first.")]}
    
    # Use enhanced schema-aware prompt
    analysis_prompt = enhance_analysis_prompt(df, question)

    try:
        response = code_llm.invoke(analysis_prompt)
        full_response = response.content if hasattr(response, 'content') else str(response)
        
        # Extract Thought
        thought = ""
        if "Thought:" in full_response:
            parts = full_response.split("Thought:")
            if len(parts) > 1:
                thought_part = parts[1]
                if "Action:" in thought_part:
                    thought = thought_part.split("Action:")[0].strip()
                else:
                    thought = thought_part.split("\n")[0].strip()
        
        # Extract Code
        code = full_response
        if "```python" in full_response:
            code = full_response.split("```python")[1].split("```")[0]
        elif "```" in full_response:
            code = full_response.split("```")[1].split("```")[0]
        
        code = '\n'.join([l.strip() for l in code.split('\n') if l.strip() and 'result' in l])
        
        # Execute
        namespace = {"df": df, "pd": pd, "np": np}
        exec(code, namespace)
        result = namespace.get("result")
        
        if result is not None:
            # Format result nicely
            if isinstance(result, (int, float)):
                formatted = f"{result:,.2f}" if isinstance(result, float) else f"{result:,}"
            else:
                formatted = str(result)
            
            thought_display = f"**🧠 Thought:** _{thought}_\n\n" if thought else ""
            
            answer = f"""{thought_display}📊 **Analysis Complete!**

**Your Question:** {question}

**Result:** {formatted}

📈 *Based on {len(df):,} total records in the dataset*"""
            
            return {"messages": [AIMessage(content=answer)]}
        else:
            return {"messages": [AIMessage(content="⚠️ Couldn't calculate that. Available numeric columns: " + ', '.join(numeric_cols))]}
            
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Analysis failed. Try: 'What's the average [column]?' or 'Count total rows'")]}

# Agent 5: Document Agent
def document_agent(state: State) -> dict:
    """Handles document queries using RAG with query rephrasing"""
    question = state["messages"][-1].content
    rag_chain = st.session_state.get("rag_chain")
    
    if rag_chain and st.session_state.vectorstore:
        # Rephrase query for better retrieval
        rephrased_queries = query_rephraser.rephrase_for_documents(question)
        
        # Show rephrased queries to user for transparency
        rephrased_display = "\n".join([f"{i+1}. {q}" for i, q in enumerate(rephrased_queries)])
        
        # Get answer from RAFT chain
        answer = rag_chain.invoke(question)
        
        # Extract Thought and Answer from RAFT response
        thought = ""
        final_answer = answer
        
        if "**Thought**:" in answer:
            parts = answer.split("**Answer**:")
            if len(parts) >= 2:
                thought_part = parts[0].replace("**Thought**:", "").strip()
                final_answer = parts[1].strip()
                thought = thought_part
        
        # Build response with rephrased queries
        response = f"🔍 **Query Variations Used:**\n{rephrased_display}\n\n"
        
        if thought:
            response += f"💭 **Analysis:** {thought}\n\n"
        
        response += final_answer
        
        return {"messages": [AIMessage(content=response)]}
    else:
        return {"messages": [AIMessage(content="❌ No documents loaded. Please upload PDF or DOCX files to ask questions about them.")]}

# NEW AGENT WRAPPERS - Use agents from agents folder

def use_cleaning_agent(state: State) -> dict:
    """Use CleaningAgent with LLM-driven code generation for complex tasks"""
    question = state["messages"][-1].content
    df = state.get("dataframes", [None])[-1] if state.get("dataframes") else None
    
    if df is None:
        return {"messages": [AIMessage(content="❌ No dataset loaded.")]}
    
    # Use enhanced prompt for cleaning
    prompt = enhance_cleaning_prompt(df, question)
    
    try:
        response = code_llm.invoke(prompt)
        full_response = response.content if hasattr(response, 'content') else str(response)
        
        # Extract Code
        code = full_response
        if "```" in full_response:
            code = full_response.split("```")[1]
            if "python" in code.split("\n")[0]:
                code = "\n".join(code.split("\n")[1:])
            code = code.strip("`").strip()
            
        # Execute
        namespace = {"df": df, "pd": pd, "np": np}
        exec(code, {}, namespace)
        result = namespace.get("result")
        
        if isinstance(result, pd.DataFrame):
            return {"messages": [AIMessage(content=f"✅ Cleaning complete! Modified {len(result)} rows.")], "filtered_df": result}
        else:
            return {"messages": [AIMessage(content=f"✅ Operation complete. Result: {result}")]}
            
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Cleaning failed: {str(e)}")]}

def use_transformation_agent(state: State) -> dict:
    """Use TransformationAgent with LLM-driven code generation"""
    question = state["messages"][-1].content
    dataframes = state.get("dataframes", [])
    
    if not dataframes:
        return {"messages": [AIMessage(content="❌ No dataset loaded.")]}
    
    df = dataframes[-1]
    
    # Use enhanced prompt for transformation (Pass ALL dataframes)
    prompt = enhance_transformation_prompt(df, question, dataframes=dataframes)
    
    try:
        response = code_llm.invoke(prompt)
        full_response = response.content if hasattr(response, 'content') else str(response)
        
        # Extract Code
        code = full_response
        if "```" in full_response:
            code = full_response.split("```")[1]
            if "python" in code.split("\n")[0]:
                code = "\n".join(code.split("\n")[1:])
            code = code.strip("`").strip()
            
        # Execute
        # CRITICAL: Pass 'dataframes' to namespace so code can access all files
        namespace = {"df": df, "dataframes": dataframes, "pd": pd, "np": np}
        exec(code, {}, namespace)
        result = namespace.get("result")
        
        if isinstance(result, pd.DataFrame):
            return {"messages": [AIMessage(content=f"✅ Transformation complete! Result has {len(result)} rows.")], "filtered_df": result}
        else:
            return {"messages": [AIMessage(content=f"✅ Operation complete. Result: {result}")]}
            
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Transformation failed: {str(e)}")]}

def use_analysis_agent_wrapper(state: State) -> dict:
    """Use AnalysisAgent from agents folder"""
    question = state["messages"][-1].content.lower()
    df = state.get("dataframes", [None])[-1] if state.get("dataframes") else None
    
    if df is None:
        return {"messages": [AIMessage(content="❌ No dataset loaded.")]}
    
    # Detect operation
    if "summary" in question or "statistics" in question:
        result = data_analysis_agent.statistical_summary(df)
    elif "correlation" in question:
        result = data_analysis_agent.correlation_analysis(df)
    elif "group" in question:
        # Example grouping
        result = {"success": True, "data": df, "message": "Analysis complete"}
    else:
        result = data_analysis_agent.statistical_summary(df)
    
    if result['success']:
        return {"messages": [AIMessage(content=f"✅ {result['message']}")], "filtered_df": result['data']}
    return {"messages": [AIMessage(content=f"❌ {result['message']}")]}

def use_visualization_agent_wrapper(state: State) -> dict:
    """Use VisualizationAgent from agents folder"""
    df = state.get("dataframes", [None])[-1] if state.get("dataframes") else None
    
    if df is None:
        return {"messages": [AIMessage(content="❌ No dataset loaded.")]}
    
    return {"messages": [AIMessage(content="📊 Visualization feature - charts will be displayed in future updates.")]}

def use_advanced_analytics_agent_wrapper(state: State) -> dict:
    """Use AdvancedAnalyticsAgent with LLM-driven code generation"""
    question = state["messages"][-1].content
    df = state.get("dataframes", [None])[-1] if state.get("dataframes") else None
    
    if df is None:
        return {"messages": [AIMessage(content="❌ No dataset loaded.")]}
    
    # Use enhanced prompt for advanced analytics
    prompt = enhance_advanced_analytics_prompt(df, question)
    
    try:
        response = code_llm.invoke(prompt)
        full_response = response.content if hasattr(response, 'content') else str(response)
        
        # Extract Code
        code = full_response
        if "```" in full_response:
            code = full_response.split("```")[1]
            if "python" in code.split("\n")[0]:
                code = "\n".join(code.split("\n")[1:])
            code = code.strip("`").strip()
            
        # Execute
        namespace = {"df": df, "pd": pd, "np": np, "KMeans": KMeans, "StandardScaler": StandardScaler}
        exec(code, {}, namespace)
        result = namespace.get("result")
        
        if isinstance(result, pd.DataFrame):
            return {"messages": [AIMessage(content=f"✅ Analytics complete! Result has {len(result)} rows.")], "filtered_df": result}
        elif isinstance(result, (pd.Series, dict, str, int, float)):
             return {"messages": [AIMessage(content=f"✅ Analysis Result:\n\n{result}")]}
        else:
            return {"messages": [AIMessage(content=f"✅ Operation complete.")]}
            
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Analytics failed: {str(e)}")]}

def use_export_agent_wrapper(state: State) -> dict:
    """Use ExportAgent from agents folder"""
    df = state.get("filtered_df") or (state.get("dataframes", [None])[-1] if state.get("dataframes") else None)
    
    if df is None:
        return {"messages": [AIMessage(content="❌ No dataset to export.")]}
    
    # Clean up: remove completely empty columns before export
    df_clean = df.dropna(axis=1, how='all')
    
    # Generate CSV
    csv_data = df_clean.to_csv(index=False)
    return {
        "messages": [AIMessage(content=f"✅ Export ready: {len(df_clean)} rows, {len(df_clean.columns)} columns")],
        "csv_files": [{"name": f"export_{len(df_clean)}_rows.csv", "data": csv_data, "rows": len(df_clean)}]
    }

# Agent 7: General Agent
# === Main Chatbot ===
def chatbot(state: State) -> dict:
    """Routes to appropriate agent based on query analysis"""
    routing_result = router_agent(state)
    agent_decision = routing_result["agent_decision"]
    
    st.session_state.last_agent = agent_decision
    
    agents = {
        "cleaning_agent": use_cleaning_agent,
        "transformation_agent": use_transformation_agent,
        "analysis_agent": use_analysis_agent_wrapper,
        "visualization_agent": use_visualization_agent_wrapper,
        "advanced_analytics_agent": use_advanced_analytics_agent_wrapper,
        "export_agent": use_export_agent_wrapper,
        "data_query_agent": data_query_agent,
        "document_agent": document_agent
    }
    
    return agents.get(agent_decision, data_query_agent)(state)

# === Graph ===
graph = StateGraph(State)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
app = graph.compile()

# ========================= LONG-TERM MEMORY =========================
CONVERSATIONS_DIR = "conversations"

def get_chat_path(chat_id):
    if chat_id is None:
        return None
    return os.path.join(CONVERSATIONS_DIR, chat_id)

def save_chat(chat_id):
    # Don't save if chat_id is None (no chat created yet)
    if chat_id is None:
        return
    
    path = get_chat_path(chat_id)
    os.makedirs(path, exist_ok=True)
    
    # Save messages
    if st.session_state.messages:
        saved_msgs = [
            {"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
            for msg in st.session_state.messages
        ]
        with open(os.path.join(path, "messages.json"), "w") as f:
            json.dump(saved_msgs, f)
    
    # Save dataframes (Multi-CSV support)
    if st.session_state.dataframes:
        # Save each dataframe with index
        for i, df in enumerate(st.session_state.dataframes):
            df.to_csv(os.path.join(path, f"dataframe_{i}.csv"), index=False)
        
        # Also save the last one as 'dataframe.csv' for backward compatibility
        st.session_state.dataframes[-1].to_csv(os.path.join(path, "dataframe.csv"), index=False)
    
    # Save filtered_df (CRITICAL for showing tables in loaded chats)
    if st.session_state.get("filtered_df") is not None:
        st.session_state.filtered_df.to_csv(os.path.join(path, "filtered_df.csv"), index=False)
    
    # Save active_df (for table display)
    if st.session_state.get("active_df") is not None:
        st.session_state.active_df.to_csv(os.path.join(path, "active_df.csv"), index=False)
    
    # Save vectorstore
    if st.session_state.vectorstore:
        st.session_state.vectorstore.save_local(os.path.join(path, "faiss_index"))

def load_chat(chat_id):
    # Don't load if chat_id is None
    if chat_id is None:
        return
    
    path = get_chat_path(chat_id)
    
    # Load messages
    msg_path = os.path.join(path, "messages.json")
    if os.path.exists(msg_path):
        with open(msg_path, "r") as f:
            saved = json.load(f)
        messages = []
        for m in saved:
            if m["type"] == "human":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))
        st.session_state.messages = messages
    else:
        st.session_state.messages = []
    
    # Load dataframes (Multi-CSV support)
    loaded_dfs = []
    
    # Try loading indexed dataframes (dataframe_0.csv, dataframe_1.csv, ...)
    i = 0
    while True:
        df_path = os.path.join(path, f"dataframe_{i}.csv")
        if os.path.exists(df_path):
            try:
                loaded_dfs.append(pd.read_csv(df_path))
                i += 1
            except Exception:
                break
        else:
            break
    
    # Fallback: if no indexed files found, try loading old 'dataframe.csv'
    if not loaded_dfs:
        legacy_path = os.path.join(path, "dataframe.csv")
        if os.path.exists(legacy_path):
            try:
                loaded_dfs.append(pd.read_csv(legacy_path))
            except Exception:
                pass
    
    st.session_state.dataframes = loaded_dfs
    
    # Load filtered_df (CRITICAL for showing tables)
    filtered_df_path = os.path.join(path, "filtered_df.csv")
    if os.path.exists(filtered_df_path):
        st.session_state.filtered_df = pd.read_csv(filtered_df_path)
        # Also set as active_df for display if not explicitly saved
        if not os.path.exists(os.path.join(path, "active_df.csv")):
            st.session_state.active_df = st.session_state.filtered_df
    else:
        st.session_state.filtered_df = None
        st.session_state.active_df = None
    
    # Load active_df (for table display)
    active_df_path = os.path.join(path, "active_df.csv")
    if os.path.exists(active_df_path):
        st.session_state.active_df = pd.read_csv(active_df_path)
    
    # Load vectorstore
    faiss_path = os.path.join(path, "faiss_index")
    if os.path.exists(faiss_path) and os.path.exists(os.path.join(faiss_path, "index.faiss")):
        st.session_state.vectorstore = FAISS.load_local(faiss_path, embed_model, allow_dangerous_deserialization=True)
        st.session_state.rag_chain = create_raft_chain(st.session_state.vectorstore)
    else:
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
    
    st.session_state.last_agent = None

def display_paginated_data():
    """Displays active dataframe with pagination"""
    if st.session_state.active_df is not None:
        with st.expander("📊 Data View (Paginated)", expanded=True):
            df = st.session_state.active_df
            rows_per_page = 10
            total_rows = len(df)
            
            if total_rows == 0:
                st.info("No data to display")
                return

            total_pages = (total_rows - 1) // rows_per_page + 1
            
            # Ensure page is valid
            if st.session_state.active_page >= total_pages:
                st.session_state.active_page = total_pages - 1
            if st.session_state.active_page < 0:
                st.session_state.active_page = 0
                
            start_idx = st.session_state.active_page * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)
            
            # Display table
            st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
            
            # Pagination Controls
            col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
            with col2:
                if st.button("◀️ Prev", key="prev_page", disabled=st.session_state.active_page == 0):
                    st.session_state.active_page -= 1
                    st.rerun()
            with col3:
                st.markdown(f"<div style='text-align: center; padding-top: 5px'>Page <b>{st.session_state.active_page + 1}</b> of {total_pages}</div>", unsafe_allow_html=True)
            with col4:
                if st.button("Next ▶️", key="next_page", disabled=st.session_state.active_page == total_pages - 1):
                    st.session_state.active_page += 1
                    st.rerun()
            
            st.caption(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
            
            # Download Button
            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False),
                f"export_{len(df)}_rows.csv",
                "text/csv",
                key="persistent_csv_download"
            )

# ========================= STREAMLIT UI =========================
st.set_page_config(page_title="Multi-Agent System", page_icon="🤖", layout="wide")
# st.title("🤖 Intelligent Multi-Agent RAG System") # Removed
# st.markdown("*Powered by 6 specialized AI agents that analyze your queries and provide intelligent responses*") # Removed

# === Session State Initialization ===
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None  # Don't create chat until first message
    st.session_state.messages = []
    st.session_state.dataframes = []
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    st.session_state.last_agent = None
    # Pagination State
    st.session_state.active_df = None
    st.session_state.active_page = 0
elif st.session_state.current_chat_id:
    # Only load if chat exists
    if os.path.exists(get_chat_path(st.session_state.current_chat_id)):
        load_chat(st.session_state.current_chat_id)

# === Sidebar ===
# === Sidebar ===
# === Sidebar Removed ===
# The sidebar functionality has been removed as requested.

    
    # === Main Chat Interface ===
        



# === Main Interface Logic ===
user_input = None

# ═══════════════════════════════════════════════════════════════════
# FINAL & PERFECT FIXED BOTTOM CHAT INPUT (2025 Streamlit)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Give space for the fixed bar */
    .main > .block-container {
        padding-bottom: 100px !important;
    }

    /* Completely hide Streamlit's default bottom bar (we'll style it ourselves) */
    section[data-testid="stBottom"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    /* Our custom fixed input bar */
    .custom-chat-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--background-color);
        border-top: 1px solid var(--secondary-background-color);
        padding: 12px 16px;
        z-index: 9999;
        box-shadow: 0 -6px 20px rgba(0,0,0,0.1);
    }

    .custom-chat-bar .inner {
        max-width: 1100px;
        margin: 0 auto;
        display: flex;
        align-items: flex-end;
        gap: 12px;
    }

    /* Attachment button - perfect circle */
    .attach-btn {
        background: var(--secondary-background-color);
        border: 1px solid #444;
        border-radius: 50%;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        flex-shrink: 0;
        font-size: 20px;
    }

    /* Chat input - clean, full width, no weird padding */
    .custom-chat-bar [data-testid="stChatInput"] {
        flex: 1;
        margin: 0 !important;
        padding: 0 !important;
    }

    .custom-chat-bar [data-testid="stChatInput"] > div {
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
    }

    .custom-chat-bar textarea {
        background: var(--secondary-background-color) !important;
        border: 1px solid #444 !important;
        border-radius: 24px !important;
        padding: 14px 56px 14px 18px !important;
        font-size: 16px !important;
        height: 48px !important;
        resize: none !important;
        box-shadow: none !important;
    }

    .custom-chat-bar textarea:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 2px rgba(0,123,255,0.25) !important;
    }

    /* Send button - perfectly centered on the right */
    .custom-chat-bar button[kind="primary"] {
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        background: var(--primary-color) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }

    .custom-chat-bar button[kind="primary"] svg {
        width: 20px !important;
        height: 20px !important;
        margin-left: 2px !important;
    }
</style>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    # === Centered Landing Page ===
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True) # Spacer
        st.markdown("<h1 style='text-align: center;'>Bailbooks AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Your Fastest Path To Defendant Data </p>", unsafe_allow_html=True)
        
        # File Uploader in Center
        uploaded_files = st.file_uploader(
            "Upload files (CSV, PDF, DOCX)", 
            type=["pdf", "docx", "doc", "csv"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        # Spacer between uploader and input
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Centered Input with Send Button
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            user_input = st.text_input(
                "Ask me anything...", 
                key="landing_input", 
                placeholder="Ask me anything...",
                label_visibility="collapsed"
            )
        with col_btn:
            # Use a form submit button look-alike or just a button
            # To make it align better, we might need some top margin or custom CSS, 
            # but standard columns are a good start.
            # Using a unicode arrow for the send icon
            submitted = st.button("➤", key="landing_submit", help="Send message")
        
        if user_input or submitted:
            # If button pressed but no input, we might want to handle that, 
            # but usually user types then presses enter OR button.
            # If button pressed, we need to grab the input value. 
            # Limitation: st.button doesn't automatically submit the text_input unless it's in a form.
            # But putting it in a form prevents 'Enter' from working nicely without 'clear_on_submit' issues sometimes.
            # Let's stick to simple logic: if user_input is present (Enter key) OR (Button pressed AND user_input is accessible).
            
            # Actually, if button is pressed, 'user_input' might be empty in this rerun if they didn't hit enter.
            # We rely on session state or just the fact that text_input preserves value on rerun if key is set.
            
            val = st.session_state.get("landing_input", "")
            if val:
                # Create chat ID if this is the first message
                if st.session_state.current_chat_id is None:
                    st.session_state.current_chat_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    os.makedirs(get_chat_path(st.session_state.current_chat_id), exist_ok=True)
                
                st.session_state.messages.append(HumanMessage(content=val))
                save_chat(st.session_state.current_chat_id) # Save before rerun to prevent data loss
                st.rerun()

else:
    # === Your normal chat messages (keep exactly as you had) ===
    st.markdown("---")
    for msg in st.session_state.messages:
        is_user = isinstance(msg, HumanMessage)
        container_class = "user-message-container" if is_user else "assistant-message-container"
        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(msg.content, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    display_paginated_data()

    # ———————————————————— ACTUAL INPUT BAR (BEAUTIFUL & FIXED) ————————————————————
    st.markdown('<div class="custom-chat-bar"><div class="inner">', unsafe_allow_html=True)

    # Attachment button (paperclip)
    col1, col2 = st.columns([1, 12])

    with col1:
        if st.button("📎", key="attach_btn", help="Upload files"):
            st.session_state.show_uploader = not st.session_state.get("show_uploader", False)

    with col2:
        prompt = st.chat_input("Type your message...", key="perfect_chat_input")

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Optional: Pop-up uploader when Attach is clicked
    if st.session_state.get("show_uploader"):
        with st.expander("📁 Upload files", expanded=True):
            uploaded_files = st.file_uploader(
                "Drop CSV, PDF, DOCX here",
                type=["csv", "pdf", "docx", "doc"],
                accept_multiple_files=True,
                key="popup_uploader"
            )
            if st.button("✕ Close", key="close_uploader"):
                st.session_state.show_uploader = False
                st.rerun()
    else:
        uploaded_files = None

    # Simple auto-scroll (optional)
    st.markdown("<script>window.scrollTo(0,document.body.scrollHeight);</script>", unsafe_allow_html=True)

    # Capture input from the new chat_input
    if prompt:
        # Create chat ID if this is the first message
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(get_chat_path(st.session_state.current_chat_id), exist_ok=True)
        
        st.session_state.messages.append(HumanMessage(content=prompt))
        user_input = prompt

# Ensure uploaded_files is defined
if 'uploaded_files' not in locals():
    uploaded_files = None

# === File Processing ===
if uploaded_files:
    with st.spinner("Processing..."):
        texts = []
        csv_count = 0
        for file in uploaded_files:
            res = parse_file(file)
            if res and res["type"] == "csv":
                df = res["data"]
                st.session_state.dataframes.append(df)
                csv_count += 1
                st.success(f"✅ Loaded CSV {csv_count}: {file.name}")
                with st.expander(f"Preview {file.name}"):
                    st.dataframe(df.head(10), use_container_width=True)
            elif res:
                texts.append(res["data"])
                st.success(f"✅ Loaded: {file.name}")

        if texts:
            chunks = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600).create_documents(texts)
            st.session_state.vectorstore = FAISS.from_documents(chunks, embed_model)
            st.session_state.rag_chain = create_raft_chain(st.session_state.vectorstore)
            st.success("✅ Documents indexed!")
        
        save_chat(st.session_state.current_chat_id)

# === Response Generation ===
if user_input or (st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage)):
    # If we just added a message (from landing or chat input), generate response
    # Check if last message is Human and we haven't responded yet (simple check)
    last_msg = st.session_state.messages[-1]
    if isinstance(last_msg, HumanMessage):
         with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            # Show thinking only briefly
            response_placeholder.markdown("💭 *Thinking...*")

            # Run the graph ONCE and get the final state
            final_state = None
            for chunk in app.stream(
                {
                    "messages": st.session_state.messages,
                    "dataframes": st.session_state.dataframes,
                    "csv_files": []
                },
                stream_mode="values"  # This is key!
            ):
                final_state = chunk  # Keep overwriting with latest state

            # Now extract ONLY the final AI message
            if final_state and "messages" in final_state:
                last_msg = final_state["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    full_response = last_msg.content

            # === NOW DO MANUAL TOKEN-BY-TOKEN STREAMING (like ChatGPT) ===
            # We'll simulate smooth typing using the final response
            import time
            response_placeholder.markdown("")  # Clear "Thinking..."

            streamed_text = ""
            if full_response:
                # Stream character by character to preserve formatting (newlines, tables, etc.)
                for char in full_response:
                    streamed_text += char
                    # Update every few characters for performance to avoid too many re-renders
                    if len(streamed_text) % 2 == 0:
                        response_placeholder.markdown(streamed_text + "●")
                        time.sleep(0.002)  # Fast typing effect
                
                # Final render without cursor
                response_placeholder.markdown(full_response)

            # === Display Data / Download Buttons ===
            should_rerun = False
            if final_state:
                if "filtered_df" in final_state and final_state["filtered_df"] is not None:
                    # Update pagination state
                    st.session_state.active_df = final_state["filtered_df"]
                    st.session_state.active_page = 0
                    should_rerun = True

                if "csv_files" in final_state and final_state["csv_files"]:
                    for csv_file in final_state["csv_files"]:
                        st.success(f"✅ Ready: {csv_file['name']}")
                        st.download_button(
                            f"📥 Download {csv_file['name']}",
                            csv_file["data"],
                            csv_file["name"],
                            "text/csv",
                            key=f"csv_dl_{uuid.uuid4()}"
                        )

                # Save final message
                if full_response:
                    st.session_state.messages.append(AIMessage(content=full_response.strip()))

            save_chat(st.session_state.current_chat_id)
            
            if should_rerun:
                st.rerun()

# === Examples ===
# Removed as per user request

