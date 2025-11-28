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

# === Models ===
llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.7,
    options={
        "num_gpu": 999,
        "num_thread": 8,
        "num_ctx": 8192,
    }
)

code_llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    options={
        "num_gpu": 999,
        "num_thread": 8,
        "num_ctx": 8192,
    }
)

embed_model = OllamaEmbeddings(model="mxbai-embed-large")

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

# === RAG Chain ===
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    template = """You are a helpful assistant. Use the context below to answer the question accurately.

Context:
{context}

Question: {question}

Provide a clear, detailed answer based on the context. If the information isn't in the context, say so clearly.

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

# ==================== AGENT SYSTEM ====================

# Agent 1: Router Agent - Decides which agent to use
def router_agent(state: State) -> dict:
    """Analyzes the user's question and routes to appropriate agent"""
    question = state["messages"][-1].content
    has_csv = len(state.get("dataframes", [])) > 0
    has_docs = st.session_state.get("vectorstore") is not None
    
    decision_prompt = f"""You are an intelligent routing system. Analyze the user's request carefully and select the best agent.

USER REQUEST: "{question}"

AVAILABLE DATA:
- CSV Dataset: {"YES" if has_csv else "NO"}
- Documents (PDF/DOCX): {"YES" if has_docs else "NO"}

AGENTS AND THEIR SPECIALTIES:

1. **csv_export_agent** - Use when user wants to:
   - Create, generate, make, or export a CSV file
   - Download data as a file
   - Save data to CSV
   - Keywords: "create csv", "export", "download", "make file", "save as csv"

2. **data_query_agent** - Use when user wants to:
   - View, show, display, or list data
   - Filter or search for specific records
   - Get specific rows or columns
   - Keywords: "show", "display", "give me", "find", "get", "list"

3. **analysis_agent** - Use when user wants to:
   - Calculate statistics (average, sum, mean, median, count)
   - Analyze trends or patterns
   - Perform mathematical operations
   - Keywords: "average", "calculate", "sum", "count", "analyze", "statistics"

4. **document_agent** - Use when user asks about:
   - Content from uploaded PDF or DOCX files
   - Information from documents
   - Summarization of documents
   - Keywords: "document", "pdf", "what does it say", "summarize"

5. **general_agent** - Use when:
   - User is having casual conversation
   - Asking general questions not related to data
   - Greeting or chatting

RULES:
- If request mentions "csv", "export", "file", "download" → csv_export_agent
- If request is about viewing/showing data → data_query_agent
- If request involves calculations → analysis_agent
- If request is about documents → document_agent
- Otherwise → general_agent

Respond with ONLY ONE of these agent names:
csv_export_agent
data_query_agent
analysis_agent
document_agent
general_agent

YOUR DECISION (one word only):"""

    try:
        response = llm.invoke(decision_prompt)
        agent_name = response.content.strip().lower() if hasattr(response, 'content') else "general_agent"
        
        # Extract agent name
        valid_agents = ['csv_export_agent', 'data_query_agent', 'analysis_agent', 'document_agent', 'general_agent']
        for agent in valid_agents:
            if agent in agent_name:
                return {"agent_decision": agent}
        
        # Fallback logic based on keywords
        q_lower = question.lower()
        if any(word in q_lower for word in ['csv', 'export', 'download', 'file', 'save', 'create file']):
            return {"agent_decision": "csv_export_agent"}
        elif any(word in q_lower for word in ['average', 'sum', 'count', 'calculate', 'analyze', 'mean']):
            return {"agent_decision": "analysis_agent"}
        elif any(word in q_lower for word in ['show', 'display', 'get', 'give', 'find', 'list']):
            return {"agent_decision": "data_query_agent"}
        elif has_docs and any(word in q_lower for word in ['document', 'pdf', 'what', 'summarize']):
            return {"agent_decision": "document_agent"}
        else:
            return {"agent_decision": "general_agent"}
    except:
        return {"agent_decision": "general_agent"}

# Agent 2: Data Query Agent - Retrieves and displays data
def data_query_agent(state: State) -> dict:
    """Handles data retrieval queries with intelligent analysis"""
    question = state["messages"][-1].content
    df = state["dataframes"][-1] if state["dataframes"] else None
    
    if df is None:
        return {"messages": [AIMessage(content="❌ No dataset loaded. Please upload a CSV file first to query data.")]}
    
    # Analyze the query first
    col_info = ", ".join(df.columns[:10])
    sample_data = df.head(3).to_string()
    
    code_prompt = f"""You are a data analysis expert. Generate precise pandas code to answer the user's query.

USER'S QUESTION: "{question}"

DATASET INFORMATION:
- Total Rows: {len(df):,}
- Columns: {col_info}

SAMPLE DATA:
{sample_data}

TASK: Write Python code to extract exactly what the user asked for.

CRITICAL INSTRUCTIONS:
1. Carefully analyze what the user wants - if they mention MULTIPLE fields (name, email, state, address), SELECT ALL OF THEM
2. Identify ALL relevant columns mentioned in the request
3. **IMPORTANT**: If user says "name", find ALL name-related columns (e.g., 'First Name', 'Last Name', 'Full Name')
4. **IMPORTANT**: If user says "address", find ALL address columns (e.g., 'Street Address', 'Address Line 1', 'Address Line 2')
5. **BE FLEXIBLE**: Handle typos and partial matches (e.g., "emal" → "Email", "nam" → "Name", "addres" → "Address")
6. Look for similar column names if exact match not found (e.g., 'CustomerEmail' for 'email', 'FullName' for 'name')
7. Write clean, working pandas code
8. Store the result in variable 'result'
9. Return a DataFrame with ALL requested columns

EXAMPLES:

Request: "show 30 defendants with their name, state, address, and email"
Available columns: ['First Name', 'Last Name', 'State', 'Street Address', 'Email']
Analysis: User wants name (First Name + Last Name), state, address (Street Address), email. Limit to 30 rows.
Code: result = df[['First Name', 'Last Name', 'State', 'Street Address', 'Email']].head(30)

Request: "give me 50 customers with name and email"
Available columns: ['CustomerFirstName', 'CustomerLastName', 'CustomerEmail']
Analysis: User wants name (both first and last) and email. Limit to 50.
Code: result = df[['CustomerFirstName', 'CustomerLastName', 'CustomerEmail']].head(50)

Request: "list users from California with their email and phone"
Analysis: Filter by state AND select email and phone columns
Code: result = df[df['State'] == 'California'][['Email', 'Phone']]

Request: "show users from California"  
Analysis: Filter by state/location column, show all columns
Code: result = df[df['State'] == 'California']

Request: "find all orders above 1000"
Analysis: Filter numeric column
Code: result = df[df['Amount'] > 1000]

Request: "list customers named John"
Analysis: Search in name column
Code: result = df[df['Name'].str.contains('John', case=False, na=False)]

Now generate code for: "{question}"

WRITE ONLY THE CODE (no explanations):"""

    try:
        response = code_llm.invoke(code_prompt)
        code = response.content if hasattr(response, 'content') else str(response)
        
        # Clean code
        code = code.strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        # Extract result line
        code_lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        code = '\n'.join([l for l in code_lines if 'result' in l])
        
        if not code:
            raise ValueError("No valid code generated")
        
        # Execute
        namespace = {"df": df, "pd": pd, "np": np}
        exec(code, namespace)
        result = namespace.get("result")
        
        if isinstance(result, pd.DataFrame) and len(result) > 0:
            # Generate natural language response
            response_prompt = f"""Generate a friendly, conversational response.

USER ASKED: "{question}"

YOU RETRIEVED: {len(result)} rows with columns: {', '.join(result.columns)}

Sample of data:
{result.head(3).to_string()}

Write a natural response that:
1. Confirms what you found
2. Gives 2-3 specific examples from the data
3. Mentions the total count
4. Is helpful and conversational like ChatGPT

Example response:
"I found 50 customer emails in the dataset! Here are a few examples:
• john.doe@example.com
• jane.smith@company.com  
• mike.wilson@email.com

I've displayed all 50 emails in the table below for you."

YOUR RESPONSE:"""

            natural_resp = llm.invoke(response_prompt)
            answer = natural_resp.content if hasattr(natural_resp, 'content') else f"Found {len(result)} matching records."
            
            return {
                "messages": [AIMessage(content=answer)],
                "filtered_df": result
            }
        else:
            return {"messages": [AIMessage(content="No matching data found for your query. Try rephrasing or check the column names.")]}
            
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
        'zip': ['zip', 'postal', 'postcode', 'zipcode']
    }
    
    # Step 1: Try keyword-based matching first
    requested_cols = []
    user_mentioned_fields = []
    
    for field, keywords in column_keywords.items():
        if any(kw in q_lower for kw in keywords):
            user_mentioned_fields.append(field)
            # Find ALL matching columns in dataframe (not just first one!)
            for col in df.columns:
                if any(kw in col.lower() for kw in keywords):
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
    
    # Enhanced prompt with strict instructions
    export_prompt = f"""You are a precise CSV export specialist.

USER REQUEST: "{question}"

DATASET COLUMNS: {', '.join(df.columns.tolist())}

TASK: Generate EXACT pandas code to export ONLY what the user asked for.

CRITICAL RULES:
- If user specifies columns (like "email", "name", "state"), SELECT ONLY THOSE
- If user says "only X, Y, Z" or "just name and email" → use ONLY those columns
- If user specifies number (e.g. "200 customers") → use .head(200) or sample(200)
- Always assign final DataFrame to variable: result
- NEVER return full df unless explicitly asked "all columns" or "everything"

EXAMPLES:

User: "create csv with 200 customers only name email and state"
→ result = df[['Name', 'Email', 'State']].head(200)
   OR if exact column names unknown:
   → Find closest matching columns!

User: "export first 100 rows with just product and price"
→ result = df[['Product Name', 'Price']].head(100)

User: "download all users from Texas"
→ result = df[df['State'] == 'Texas']

User: "give me 50 random emails"
→ result = df['Email'].sample(50).to_frame()

Now generate precise code for:
"{question}"

Return ONLY valid Python code. No explanations."""
    
    try:
        response = code_llm.invoke(export_prompt)
        raw_code = response.content
        
        # Better code extraction
        code = raw_code.strip()
        if "```" in code:
            code = code.split("```")[1]
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
            
        # FINAL VALIDATION: Did we actually select the right columns?
        requested_cols_hint = re.findall(r'(?:only|just|with)\s+([a-zA-Z\s,_&and]+)', question.lower())
        if requested_cols_hint:
            hint = requested_cols_hint[0].lower()
            selected = set([col.lower() for col in result.columns])
            expected_keywords = [word.strip() for word in hint.replace(" and ", ",").split(",") if word.strip()]
            
            if not any(kw in " ".join(selected) for kw in expected_keywords):
                st.warning(f"Note: Exported columns: {', '.join(result.columns)}. Make sure spelling matches your data.")

        csv_data = result.to_csv(index=False)
        filename = f"export_{len(result)}_records.csv"
        
        answer = f"""**CSV Export Ready!**

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
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    analysis_prompt = f"""You are a data analyst. Generate code to calculate what the user asked for.

USER REQUEST: "{question}"

NUMERIC COLUMNS: {', '.join(numeric_cols) if numeric_cols else "None"}
ALL COLUMNS: {', '.join(df.columns)}
TOTAL ROWS: {len(df):,}

EXAMPLES:

Request: "what's the average age"
Code: result = df['Age'].mean()

Request: "total sales amount"
Code: result = df['Sales'].sum()

Request: "count how many customers"
Code: result = len(df)

Request: "maximum price"
Code: result = df['Price'].max()

Generate code for: "{question}"

CODE:"""

    try:
        response = code_llm.invoke(analysis_prompt)
        code = response.content if hasattr(response, 'content') else str(response)
        
        # Clean
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
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
            
            answer = f"""📊 **Analysis Complete!**

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
    """Handles document queries using RAG"""
    question = state["messages"][-1].content
    rag_chain = st.session_state.get("rag_chain")
    
    if rag_chain and st.session_state.vectorstore:
        answer = rag_chain.invoke(question)
        return {"messages": [AIMessage(content=answer)]}
    else:
        return {"messages": [AIMessage(content="❌ No documents loaded. Please upload PDF or DOCX files to ask questions about them.")]}

# Agent 6: General Agent
def general_agent(state: State) -> dict:
    """Handles general conversation"""
    response = llm.invoke(state["messages"])
    answer = response.content if hasattr(response, 'content') else str(response)
    return {"messages": [AIMessage(content=answer)]}

# === Main Chatbot ===
def chatbot(state: State) -> dict:
    """Routes to appropriate agent based on query analysis"""
    routing_result = router_agent(state)
    agent_decision = routing_result["agent_decision"]
    
    st.session_state.last_agent = agent_decision
    
    agents = {
        "data_query_agent": data_query_agent,
        "csv_export_agent": csv_export_agent,
        "analysis_agent": analysis_agent,
        "document_agent": document_agent,
        "general_agent": general_agent
    }
    
    return agents.get(agent_decision, general_agent)(state)

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
    
    # Save dataframe
    if st.session_state.dataframes:
        st.session_state.dataframes[-1].to_csv(os.path.join(path, "dataframe.csv"), index=False)
    
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
    
    # Load dataframe
    df_path = os.path.join(path, "dataframe.csv")
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
        st.session_state.dataframes = [df]
    else:
        st.session_state.dataframes = []
    
    # Load vectorstore
    faiss_path = os.path.join(path, "faiss_index")
    if os.path.exists(faiss_path) and os.path.exists(os.path.join(faiss_path, "index.faiss")):
        st.session_state.vectorstore = FAISS.load_local(faiss_path, embed_model, allow_dangerous_deserialization=True)
        st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
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
with st.sidebar:
    
    if st.button("🗑️ Clear All Chats"):
        # 1. Clear Session State FIRST to release file handles
        st.session_state.messages = []
        st.session_state.dataframes = []
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.last_agent = None
        st.session_state.active_df = None
        st.session_state.active_page = 0
        
        # Force garbage collection to release file locks (crucial for Windows)
        import gc
        gc.collect()
        
        # 2. Define error handler for Windows read-only files
        def remove_readonly(func, path, excinfo):
            os.chmod(path, 0o777)
            func(path)
            
        # 3. Clear ALL conversations with robust error handling
        if os.path.exists(CONVERSATIONS_DIR):
            try:
                shutil.rmtree(CONVERSATIONS_DIR, onerror=remove_readonly)
            except Exception as e:
                st.error(f"Error clearing chats: {e}")
                
        os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
        
        # Reset current chat ID
        st.session_state.current_chat_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(get_chat_path(st.session_state.current_chat_id), exist_ok=True)
        
        st.rerun()
    
    st.markdown("---")
    st.subheader("🤖 AI Agents")
    agents_list = [
        "🧭 Router (Analyzes queries)",
        "🔍 Data Query (Shows data)",
        "📥 CSV Export (Creates files)",
        "📊 Analysis (Calculations)",
        "📄 Document (PDF/DOCX)",
        "💬 General (Conversation)"
    ]
    for agent in agents_list:
        st.text(agent)
    
    if st.session_state.last_agent:
        st.success(f"**Active:** {st.session_state.last_agent.replace('_', ' ').title()}")
    
    if st.session_state.dataframes:
        st.markdown("---")
        df = st.session_state.dataframes[-1]
        st.metric("📊 Rows", f"{len(df):,}")
        st.metric("📋 Columns", len(df.columns))
    
    st.markdown("---")
    st.subheader("💬 Chats")
    # Only show New Chat button if there's an active chat
    if st.session_state.current_chat_id is not None:
        if st.button("➕ New Chat"):
            save_chat(st.session_state.current_chat_id)  # Save current before new
            st.session_state.current_chat_id = None  # Don't create until first message
            st.session_state.messages = []
            st.session_state.dataframes = []
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.last_agent = None
            st.session_state.active_df = None
            st.session_state.active_page = 0
            st.rerun()
    
    # List previous chats (only show non-empty chats)
    chat_dirs = [d for d in os.listdir(CONVERSATIONS_DIR) if os.path.isdir(os.path.join(CONVERSATIONS_DIR, d))]
    for cid in sorted(chat_dirs, reverse=True):  # Newest first
        msg_path = os.path.join(CONVERSATIONS_DIR, cid, "messages.json")
        
        # Only show chats that have messages
        if os.path.exists(msg_path):
            with open(msg_path, "r") as f:
                try:
                    msgs = json.load(f)
                except:
                    msgs = []
            
            # Skip empty chats
            if not msgs:
                continue
                
            # Generate proper title from first user message
            first_content = msgs[0]["content"]
            name = first_content[:40] + "..." if len(first_content) > 40 else first_content
            
            # Only show if not the current chat
            if cid != st.session_state.current_chat_id:
                if st.button(name, key=f"chat_{cid}"):
                    save_chat(st.session_state.current_chat_id)  # Save current
                    st.session_state.current_chat_id = cid
                    load_chat(cid)
                    st.rerun()
    
    # === Data Analysis Operations ===
    if st.session_state.dataframes:
        st.markdown("---")
        st.subheader("🔧 Data Analysis Tools")
        
        df = st.session_state.dataframes[-1]
        
        # Category tabs
        analysis_category = st.selectbox(
            "Select Operation Category:",
            ["Data Cleaning", "Data Transformation", "Data Analysis", 
             "Visualization", "Export", "Advanced Analytics"],
            key="analysis_category"
        )
        
        if analysis_category == "Data Cleaning":
            st.markdown("**🧹 Cleaning Operations**")
            operation = st.selectbox(
                "Operation:",
                ["Remove Duplicates", "Fix Email Formatting", "Fix Phone Formatting", 
                 "Fix Name Formatting", "Trim Whitespace", "Normalize Case",
                 "Handle Missing Values", "Standardize Columns"],
                key="cleaning_op"
            )
            
            if operation == "Remove Duplicates":
                if st.button("🧹 Remove Duplicates", key="clean_dupes"):
                    result = cleaning_agent.remove_duplicates(df)
                    if result['success']:
                        st.session_state.dataframes[-1] = result['data']
                        st.success(result['message'])
                        st.rerun()
            
            elif operation == "Trim Whitespace":
                if st.button("✂️ Trim Whitespace", key="clean_whitespace"):
                    result = cleaning_agent.trim_whitespace(df)
                    if result['success']:
                        st.session_state.dataframes[-1] = result['data']
                        st.success(result['message'])
                        st.rerun()
            
            elif operation == "Standardize Columns":
                if st.button("📝 Standardize Column Names", key="clean_cols"):
                    result = cleaning_agent.standardize_columns(df)
                    if result['success']:
                        st.session_state.dataframes[-1] = result['data']
                        st.success(result['message'])
                        st.info(f"Renamed: {list(result['metadata']['column_mapping'].items())[:3]}")
                        st.rerun()
        
        elif analysis_category == "Data Analysis":
            st.markdown("**📊 Analysis Operations**")
            operation = st.selectbox(
                "Operation:",
                ["Statistical Summary", "Value Counts", "Correlation Analysis"],
                key="analysis_op"
            )
            
            if operation == "Statistical Summary":
                if st.button("📊 Generate Summary", key="stat_summary"):
                    result = data_analysis_agent.statistical_summary(df)
                    if result['success']:
                        st.dataframe(result['data'])
                        with st.expander("📋 Detailed Stats"):
                            st.json(result['metadata']['detailed_summary'])
            
            elif operation == "Value Counts":
                column = st.selectbox("Select Column:", df.columns.tolist(), key="vc_col")
                top_n = st.number_input("Top N:", min_value=5, value=10, key="vc_n")
                if st.button("🔢 Count Values", key="value_counts"):
                    result = data_analysis_agent.value_counts(df, column, int(top_n))
                    if result['success']:
                        st.dataframe(result['data'])
        
        elif analysis_category == "Visualization":
            st.markdown("**📈 Visualization**")
            chart_type = st.selectbox(
                "Chart Type:",
                ["Bar Chart", "Pie Chart", "Histogram", "Heatmap"],
                key="viz_type"
            )
            
            if chart_type == "Bar Chart":
                x_col = st.selectbox("X-axis:", df.columns.tolist(), key="bar_x")
                if st.button("📊 Create Bar Chart", key="create_bar"):
                    result = visualization_agent.create_bar_chart(df, x_col, interactive=False)
                    if result['success'] and result['data'].get('type') == 'matplotlib':
                        import base64
                        img_data = base64.b64decode(result['data']['image_base64'])
                        st.image(img_data)
            
            elif chart_type == "Histogram":
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    col = st.selectbox("Column:", numeric_cols, key="hist_col")
                    if st.button("📈 Create Histogram", key="create_hist"):
                        result = visualization_agent.create_histogram(df, col, interactive=False)
                        if result['success'] and result['data'].get('type') == 'matplotlib':
                            import base64
                            img_data = base64.b64decode(result['data']['image_base64'])
                            st.image(img_data)
        
        elif analysis_category == "Export":
            st.markdown("**💾 Export Operations**")
            export_format = st.selectbox(
                "Format:",
                ["CSV", "Excel", "JSON"],
                key="export_format"
            )
            filename = st.text_input("Filename:", value="export", key="export_filename")
            
            if st.button(f"💾 Export to {export_format}", key="do_export"):
                file_path = f"exports/{filename}.{export_format.lower()}"
                
                if export_format == "CSV":
                    result = export_agent.export_to_csv(df, file_path)
                elif export_format == "Excel":
                    result = export_agent.export_to_excel(df, file_path)
                elif export_format == "JSON":
                    result = export_agent.export_to_json(df, file_path)
                
                if result['success']:
                    st.success(result['message'])
                    try:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                f"⬇️ Download {export_format}",
                                f,
                                file_name=f"{filename}.{export_format.lower()}",
                                key=f"download_{export_format}"
                            )
                    except:
                        pass
        
        elif analysis_category == "Advanced Analytics":
            st.markdown("**🎯 Advanced Operations**")
            operation = st.selectbox(
                "Operation:",
                ["Detect Outliers", "Validate Emails", "Validate Phones"],
                key="advanced_op"
            )
            
            if operation == "Detect Outliers":
                method = st.selectbox("Method:", ["iqr", "zscore"], key="outlier_method")
                if st.button("🔍 Detect Outliers", key="detect_outliers"):
                    result = advanced_analytics_agent.detect_outliers(df, method=method)
                    if result['success']:
                        st.dataframe(result['data'])
                        st.info(f"Total outlier rows: {result['metadata']['total_outlier_rows']}")
            
            elif operation == "Validate Emails":
                email_cols = [col for col in df.columns if 'email' in col.lower() or 'mail' in col.lower()]
                if email_cols:
                    col = st.selectbox("Email Column:", email_cols, key="email_col")
                    if st.button("✅ Validate Emails", key="validate_emails"):
                        result = advanced_analytics_agent.validate_emails(df, col)
                        if result['success']:
                            st.dataframe(result['data'])
                            st.info(f"Valid: {result['metadata']['valid_count']}, Invalid: {result['metadata']['invalid_count']}")


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
        for file in uploaded_files:
            res = parse_file(file)
            if res and res["type"] == "csv":
                df = res["data"]
                st.session_state.dataframes = [df]
                st.success(f"✅ Loaded: {file.name}")
                with st.expander(f"Preview {file.name}"):
                    st.dataframe(df.head(10), use_container_width=True)
            elif res:
                texts.append(res["data"])
                st.success(f"✅ Loaded: {file.name}")

        if texts:
            chunks = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600).create_documents(texts)
            st.session_state.vectorstore = FAISS.from_documents(chunks, embed_model)
            st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
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

