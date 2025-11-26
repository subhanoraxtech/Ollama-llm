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

# === Models ===
llm = ChatOllama(
    model="gpt-oss",
    temperature=0.7,
    options={
        "num_gpu": 999,
        "num_thread": 8,
        "num_ctx": 8192,
    }
)

code_llm = ChatOllama(
    model="gpt-oss",
    temperature=0,
    options={
        "num_gpu": 999,
        "num_thread": 8,
        "num_ctx": 8192,
    }
)

embed_model = OllamaEmbeddings(model="mxbai-embed-large")

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

INSTRUCTIONS:
1. Carefully analyze what the user wants
2. Identify the relevant columns (look for similar names if exact match not found)
3. Write clean, working pandas code
4. Store the result in variable 'result'
5. Return a DataFrame when selecting/filtering data

EXAMPLES:

Request: "give me 50 customer emails"
Analysis: User wants email addresses, limit to 50
Code: result = df[['Email']].head(50)

Request: "show users from California"  
Analysis: Filter by state/location column
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

def smart_fallback_query(df: pd.DataFrame, question: str) -> dict:
    """Intelligent fallback when code generation fails"""
    q_lower = question.lower()
    
    # Extract number from query
    numbers = re.findall(r'\d+', question)
    n = int(numbers[0]) if numbers else 10
    
    # Pattern matching
    if 'email' in q_lower:
        for col in df.columns:
            if 'email' in col.lower():
                result = df[[col]].head(n)
                examples = '\n'.join([f"• {email}" for email in result[col].head(3)])
                answer = f"Found {len(result)} emails:\n\n{examples}\n\nShowing all in the table below."
                return {"messages": [AIMessage(content=answer)], "filtered_df": result}
    
    # Default: show first N rows
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
    return os.path.join(CONVERSATIONS_DIR, chat_id)

def save_chat(chat_id):
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
    st.session_state.current_chat_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(get_chat_path(st.session_state.current_chat_id), exist_ok=True)
    st.session_state.messages = []
    st.session_state.dataframes = []
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None
    st.session_state.last_agent = None
    # Pagination State
    st.session_state.active_df = None
    st.session_state.active_page = 0
else:
    # Ensure the chat is loaded (in case of rerun)
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
    if st.button("➕ New Chat"):
        save_chat(st.session_state.current_chat_id)  # Save current before new
        new_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.current_chat_id = new_id
        os.makedirs(get_chat_path(new_id), exist_ok=True)
        st.session_state.messages = []
        st.session_state.dataframes = []
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.last_agent = None
        st.session_state.active_df = None
        st.session_state.active_page = 0
        st.rerun()
    
    # List previous chats
    chat_dirs = [d for d in os.listdir(CONVERSATIONS_DIR) if os.path.isdir(os.path.join(CONVERSATIONS_DIR, d))]
    for cid in sorted(chat_dirs, reverse=True):  # Newest first
        msg_path = os.path.join(CONVERSATIONS_DIR, cid, "messages.json")
        name = cid.replace("_", " ").replace("-", "/")
        if os.path.exists(msg_path):
            with open(msg_path, "r") as f:
                try:
                    msgs = json.load(f)
                except:
                    msgs = []
            if msgs:
                first_content = msgs[0]["content"]
                name = first_content[:30] + "..." if len(first_content) > 30 else first_content
        if cid != st.session_state.current_chat_id:
            if st.button(name, key=f"chat_{cid}"):
                save_chat(st.session_state.current_chat_id)  # Save current
                st.session_state.current_chat_id = cid
                load_chat(cid)
                st.rerun()

# === Main Interface Logic ===
user_input = None

# Custom CSS for Centering and Chat Alignment
st.markdown("""
<style>
    .center-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
    }
    .center-text {
        text-align: center;
    }
    
    /* User message container - RIGHT SIDE */
    .user-message-container {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
        margin-bottom: 1rem !important;
    }
    
    .user-message-container [data-testid="stChatMessage"] {
        max-width: 75% !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        background-color: #007bff20 !important;
        border-radius: 15px !important;
        padding: 10px 15px !important;
    }
    
    /* Assistant message container - LEFT SIDE */
    .assistant-message-container {
        display: flex !important;
        justify-content: flex-start !important;
        width: 100% !important;
        margin-bottom: 1rem !important;
    }
    
    .assistant-message-container [data-testid="stChatMessage"] {
        max-width: 75% !important;
        margin-left: 0 !important;
        margin-right: auto !important;
        background-color: #2d2d2d !important;
        border-radius: 15px !important;
        padding: 10px 15px !important;
    }
    
    /* Upload button styling */
    [data-testid="stPopover"] button {
        height: 45px !important;
        border-radius: 10px !important;
        font-size: 24px !important;
        background-color: #1f1f1f !important;
        border: 1px solid #444 !important;
    }
    
    [data-testid="stPopover"] button:hover {
        background-color: #2d2d2d !important;
        border-color: #666 !important;
    }
    
    /* Input wrapper to keep elements together at bottom */
    .input-wrapper {
        position: sticky !important;
        bottom: 0 !important;
        background-color: #0e1117 !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        z-index: 100 !important;
        border-top: 1px solid #262730 !important;
    }
    
    /* Make popover button match chat input height */
    .input-wrapper [data-testid="stPopover"] button {
        height: 50px !important;
        min-height: 50px !important;
        border-radius: 12px !important;
        font-size: 22px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Remove expander styles since we're using popover now */
    [data-testid="stExpander"] {
        border: none !important;
    }
    
    /* Ensure chat input stays at bottom */
    [data-testid="stChatInput"] {
        margin-bottom: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    # === Centered Landing Page ===
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True) # Spacer
        st.markdown("<h1 style='text-align: center;'>What can I help with?</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Multi-Agent RAG System</p>", unsafe_allow_html=True)
        
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
                st.session_state.messages.append(HumanMessage(content=val))
                save_chat(st.session_state.current_chat_id) # Save before rerun to prevent data loss
                st.rerun()

else:
    # === Standard Chat View ===
    st.markdown("---")
    
    for msg in st.session_state.messages:
        is_user = isinstance(msg, HumanMessage)
        # Wrap in custom container for styling
        container_class = "user-message-container" if is_user else "assistant-message-container"
        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(msg.content)
        st.markdown('</div>', unsafe_allow_html=True)

    display_paginated_data()
    
    # Create a container for the input area
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    
    # Place upload button and chat input on the same line
    col_upload, col_input = st.columns([0.06, 0.94])
    
    with col_upload:
        # Compact upload button using popover
        with st.popover("➕", use_container_width=True):
            st.markdown("**📎 Upload Files**")
            uploaded_files = st.file_uploader(
                "Add files",
                type=["pdf", "docx", "doc", "csv"],
                accept_multiple_files=True,
                key="chat_uploader",
                label_visibility="collapsed"
            )
    
    with col_input:
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            st.session_state.messages.append(HumanMessage(content=prompt))
            user_input = prompt
    
    st.markdown('</div>', unsafe_allow_html=True)

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

