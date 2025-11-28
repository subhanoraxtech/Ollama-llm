# ==================== COMPREHENSIVE AGENT WRAPPERS ====================
# This file contains all agent wrapper functions for the multi-agent system
# Copy these functions into app.py after the router_agent function

def get_target_dataframe(state: State, question: str):
    """Helper function to determine which CSV to operate on based on user query"""
    dataframes = state.get("dataframes", [])
    csv_metadata = state.get("csv_metadata", [])
    q_lower = question.lower()
    
    if not dataframes:
        return None, None, "No CSV loaded"
    
    # Check for specific CSV by name
    for i, meta in enumerate(csv_metadata):
        csv_name = meta.get('name', '').lower()
        if csv_name and csv_name in q_lower:
            return dataframes[i], meta.get('name', f'CSV{i+1}'), None
    
    # Check for index-based selection
    if 'first' in q_lower and len(dataframes) > 0:
        return dataframes[0], csv_metadata[0].get('name', 'first CSV') if csv_metadata else 'first CSV', None
    elif 'second' in q_lower and len(dataframes) > 1:
        return dataframes[1], csv_metadata[1].get('name', 'second CSV') if len(csv_metadata) > 1 else 'second CSV', None
    elif 'third' in q_lower and len(dataframes) > 2:
        return dataframes[2], csv_metadata[2].get('name', 'third CSV') if len(csv_metadata) > 2 else 'third CSV', None
    elif 'last' in q_lower:
        return dataframes[-1], csv_metadata[-1].get('name', 'last CSV') if csv_metadata else 'last CSV', None
    
    # Default to last uploaded
    return dataframes[-1], csv_metadata[-1].get('name', 'latest CSV') if csv_metadata else 'latest CSV', None


def cleaning_agent_wrapper(state: State) -> dict:
    """Handles data cleaning operations using CleaningAgent class"""
    question = state["messages"][-1].content
    q_lower = question.lower()
    
    # Get target dataframe
    df, df_name, error = get_target_dataframe(state, question)
    if error:
        return {"messages": [AIMessage(content=f"❌ {error}. Please upload CSV files first.")]}
    
    # Detect cleaning operation
    if 'duplicate' in q_lower or 'duplicates' in q_lower:
        result = cleaning_agent.remove_duplicates(df)
        if result['success']:
            # Update the dataframe in state
            idx = state["dataframes"].index(df)
            state["dataframes"][idx] = result['data']
            return {
                "messages": [AIMessage(content=f"✅ **{df_name}**: {result['message']}")],
                "dataframes": state["dataframes"],
                "filtered_df": result['data']
            }
        else:
            return {"messages": [AIMessage(content=f"❌ {result['message']}")]}
    
    elif 'email' in q_lower and ('fix' in q_lower or 'clean' in q_lower or 'format' in q_lower):
        # Find email column
        email_cols = [col for col in df.columns if 'email' in col.lower() or 'mail' in col.lower()]
        if email_cols:
            result = cleaning_agent.fix_email_formatting(df, email_cols[0])
            if result['success']:
                idx = state["dataframes"].index(df)
                state["dataframes"][idx] = result['data']
                return {
                    "messages": [AIMessage(content=f"✅ **{df_name}**: {result['message']}")],
                    "dataframes": state["dataframes"],
                    "filtered_df": result['data']
                }
        return {"messages": [AIMessage(content="❌ No email column found.")]}
    
    elif 'phone' in q_lower and ('fix' in q_lower or 'clean' in q_lower or 'format' in q_lower):
        phone_cols = [col for col in df.columns if 'phone' in col.lower()]
        if phone_cols:
            result = cleaning_agent.fix_phone_formatting(df, phone_cols[0])
            if result['success']:
                idx = state["dataframes"].index(df)
                state["dataframes"][idx] = result['data']
                return {
                    "messages": [AIMessage(content=f"✅ **{df_name}**: {result['message']}")],
                    "dataframes": state["dataframes"],
                    "filtered_df": result['data']
                }
        return {"messages": [AIMessage(content="❌ No phone column found.")]}
    
    elif 'whitespace' in q_lower or 'trim' in q_lower:
        result = cleaning_agent.trim_whitespace(df)
        if result['success']:
            idx = state["dataframes"].index(df)
            state["dataframes"][idx] = result['data']
            return {
                "messages": [AIMessage(content=f"✅ **{df_name}**: {result['message']}")],
                "dataframes": state["dataframes"],
                "filtered_df": result['data']
            }
    
    elif 'missing' in q_lower or 'null' in q_lower or 'nan' in q_lower:
        result = cleaning_agent.handle_missing_values(df, strategy='drop')
        if result['success']:
            idx = state["dataframes"].index(df)
            state["dataframes"][idx] = result['data']
            return {
                "messages": [AIMessage(content=f"✅ **{df_name}**: {result['message']}")],
                "dataframes": state["dataframes"],
                "filtered_df": result['data']
            }
    
    elif 'standardize' in q_lower and 'column' in q_lower:
        result = cleaning_agent.standardize_columns(df)
        if result['success']:
            idx = state["dataframes"].index(df)
            state["dataframes"][idx] = result['data']
            return {
                "messages": [AIMessage(content=f"✅ **{df_name}**: {result['message']}")],
                "dataframes": state["dataframes"],
                "filtered_df": result['data']
            }
    
    # List available operations
    ops = cleaning_agent.get_available_operations()
    ops_list = "\n".join([f"- **{k}**: {v}" for k, v in ops.items()])
    return {"messages": [AIMessage(content=f"""I can perform these cleaning operations on **{df_name}**:

{ops_list}

Try: "remove duplicates", "fix email formatting", "trim whitespace", etc.""")]}


def analysis_agent_wrapper(state: State) -> dict:
    """Handles statistical analysis using AnalysisAgent class"""
    question = state["messages"][-1].content
    q_lower = question.lower()
    
    # Get target dataframe
    df, df_name, error = get_target_dataframe(state, question)
    if error:
        return {"messages": [AIMessage(content=f"❌ {error}. Please upload CSV files first.")]}
    
    # Statistical summary
    if 'summary' in q_lower or 'statistics' in q_lower or 'stats' in q_lower:
        result = data_analysis_agent.statistical_summary(df)
        if result['success']:
            summary_df = result['data']
            return {
                "messages": [AIMessage(content=f"✅ **Statistical Summary for {df_name}**:\n\n{result['message']}")],
                "filtered_df": summary_df
            }
    
    # Correlation analysis
    elif 'correlation' in q_lower or 'correlate' in q_lower:
        result = data_analysis_agent.correlation_analysis(df)
        if result['success']:
            return {
                "messages": [AIMessage(content=f"✅ **Correlation Analysis for {df_name}**:\n\n{result['message']}")],
                "filtered_df": result['data']
            }
    
    # Value counts
    elif 'count' in q_lower or 'frequency' in q_lower:
        # Try to detect column name
        for col in df.columns:
            if col.lower() in q_lower:
                result = data_analysis_agent.value_counts(df, col)
                if result['success']:
                    return {
                        "messages": [AIMessage(content=f"✅ **Value Counts for {col} in {df_name}**:\n\n{result['message']}")],
                        "filtered_df": result['data']
                    }
        return {"messages": [AIMessage(content="❌ Please specify which column to count values for.")]}
    
    # Compare datasets
    elif 'compare' in q_lower and len(state["dataframes"]) >= 2:
        df1 = state["dataframes"][0]
        df2 = state["dataframes"][1]
        result = data_analysis_agent.compare_datasets(df1, df2)
        if result['success']:
            return {"messages": [AIMessage(content=f"✅ **Dataset Comparison**:\n\n{result['message']}")]}
    
    # List available operations
    ops = data_analysis_agent.get_available_operations()
    ops_list = "\n".join([f"- **{k}**: {v}" for k, v in ops.items()])
    return {"messages": [AIMessage(content=f"""I can perform these analyses on **{df_name}**:

{ops_list}

Try: "show statistics", "correlation analysis", "count values in [column]", etc.""")]}


def visualization_agent_wrapper(state: State) -> dict:
    """Handles data visualization using VisualizationAgent class"""
    question = state["messages"][-1].content
    q_lower = question.lower()
    
    # Get target dataframe
    df, df_name, error = get_target_dataframe(state, question)
    if error:
        return {"messages": [AIMessage(content=f"❌ {error}. Please upload CSV files first.")]}
    
    # Detect chart type
    if 'bar' in q_lower or 'bar chart' in q_lower:
        # Try to detect column
        for col in df.columns:
            if col.lower() in q_lower:
                result = visualization_agent.create_bar_chart(df, col, interactive=False)
                if result['success'] and result['data'].get('type') == 'matplotlib':
                    import base64
                    img_data = base64.b64decode(result['data']['image_base64'])
                    # Save to temp file for display
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        tmp.write(img_data)
                        tmp_path = tmp.name
                    return {
                        "messages": [AIMessage(content=f"✅ **Bar Chart for {col} in {df_name}**")],
                        "csv_files": [{"type": "image", "path": tmp_path}]
                    }
        return {"messages": [AIMessage(content="❌ Please specify which column to visualize.")]}
    
    elif 'histogram' in q_lower:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            if col.lower() in q_lower:
                result = visualization_agent.create_histogram(df, col, interactive=False)
                if result['success']:
                    return {"messages": [AIMessage(content=f"✅ **Histogram for {col} in {df_name}**")]}
        return {"messages": [AIMessage(content=f"❌ Available numeric columns: {', '.join(numeric_cols)}")]}
    
    # List available operations
    ops = visualization_agent.get_available_operations()
    ops_list = "\n".join([f"- **{k}**: {v}" for k, v in ops.items()])
    return {"messages": [AIMessage(content=f"""I can create these visualizations for **{df_name}**:

{ops_list}

Try: "create bar chart for [column]", "show histogram of [column]", etc.""")]}


def advanced_analytics_agent_wrapper(state: State) -> dict:
    """Handles advanced analytics using AdvancedAnalyticsAgent class"""
    question = state["messages"][-1].content
    q_lower = question.lower()
    
    # Get target dataframe
    df, df_name, error = get_target_dataframe(state, question)
    if error:
        return {"messages": [AIMessage(content=f"❌ {error}. Please upload CSV files first.")]}
    
    # Detect outliers
    if 'outlier' in q_lower or 'anomal' in q_lower:
        method = 'iqr' if 'iqr' in q_lower else 'zscore'
        result = advanced_analytics_agent.detect_outliers(df, method=method)
        if result['success']:
            return {
                "messages": [AIMessage(content=f"✅ **Outlier Detection for {df_name}**:\n\n{result['message']}\n\nTotal outlier rows: {result['metadata']['total_outlier_rows']}")],
                "filtered_df": result['data']
            }
    
    # Validate emails
    elif 'validate' in q_lower and 'email' in q_lower:
        email_cols = [col for col in df.columns if 'email' in col.lower()]
        if email_cols:
            result = advanced_analytics_agent.validate_emails(df, email_cols[0])
            if result['success']:
                return {
                    "messages": [AIMessage(content=f"✅ **Email Validation for {df_name}**:\n\nValid: {result['metadata']['valid_count']}, Invalid: {result['metadata']['invalid_count']}")],
                    "filtered_df": result['data']
                }
        return {"messages": [AIMessage(content="❌ No email column found.")]}
    
    # List available operations
    ops = advanced_analytics_agent.get_available_operations()
    ops_list = "\n".join([f"- **{k}**: {v}" for k, v in ops.items()])
    return {"messages": [AIMessage(content=f"""I can perform these advanced operations on **{df_name}**:

{ops_list}

Try: "detect outliers", "validate emails", etc.""")]}
