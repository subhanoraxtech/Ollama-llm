# Agent 7: Multi-CSV Agent - Handles operations on multiple CSVs
def multi_csv_agent(state: State) -> dict:
    """Intelligently handles multi-CSV operations based on user request"""
    question = state["messages"][-1].content
    dataframes = state.get("dataframes", [])
    
    if len(dataframes) < 2:
        return {"messages": [AIMessage(content="❌ Please upload at least 2 CSV files to perform multi-CSV operations.")]}
    
    # Use LLM to understand what operation user wants
    operation_prompt = f"""Analyze the user's request and determine what multi-CSV operation they want.

USER REQUEST: "{question}"

AVAILABLE OPERATIONS:
1. **compare_structure** - Compare columns, data types, shapes
2. **compare_data** - Compare actual data values, find mismatches
3. **merge** - Join/merge CSVs (inner, left, right, outer, cross join)
4. **find_differences** - Find new, deleted, modified rows
5. **get_statistics** - Get summary statistics for comparison

KEYWORDS:
- compare, difference, different → compare_structure or compare_data
- merge, join, combine → merge
- changes, new, deleted, modified → find_differences
- stats, statistics, summary → get_statistics

Respond with ONLY the operation name (one word):"""

    try:
        response = llm.invoke(operation_prompt)
        operation = response.content.strip().lower() if hasattr(response, 'content') else "compare_structure"
        
        df1 = dataframes[0]
        df2 = dataframes[1] if len(dataframes) > 1 else dataframes[0]
        
        # Execute the appropriate operation
        if 'compare_structure' in operation or 'structure' in operation:
            result = csv_comparator.compare_structure(df1, df2, "File 1", "File 2")
            
            answer = f"""📊 **Structure Comparison**

**File 1**: {result['shape1'][0]} rows × {result['shape1'][1]} columns
**File 2**: {result['shape2'][0]} rows × {result['shape2'][1]} columns

**Common Columns** ({len(result['common_columns'])}): {', '.join(result['common_columns'][:10])}
**Only in File 1** ({len(result['only_in_1'])}): {', '.join(result['only_in_1'][:5]) if result['only_in_1'] else 'None'}
**Only in File 2** ({len(result['only_in_2'])}): {', '.join(result['only_in_2'][:5]) if result['only_in_2'] else 'None'}

**Type Mismatches**: {len(result['type_mismatches'])} columns have different types

✅ **Structure Identical**: {result['structure_identical']}"""
            
            return {"messages": [AIMessage(content=answer)]}
        
        elif 'compare_data' in operation or 'data' in operation:
            # Try to find a key column
            common_cols = list(set(df1.columns) & set(df2.columns))
            key_col = None
            for col in common_cols:
                if 'id' in col.lower() or 'key' in col.lower():
                    key_col = col
                    break
            
            result = csv_comparator.compare_data(df1, df2, key_col, "File 1", "File 2")
            
            if not result['success']:
                return {"messages": [AIMessage(content=f"❌ {result.get('error', 'Comparison failed')}")]}
            
            answer = f"""🔍 **Data Comparison**

**File 1**: {result['total_rows_1']} rows
**File 2**: {result['total_rows_2']} rows
"""
            
            if key_col:
                answer += f"""
**Key Column**: {key_col}
**Rows only in File 1**: {result.get('rows_only_in_1', 0)}
**Rows only in File 2**: {result.get('rows_only_in_2', 0)}
**Common Rows**: {result.get('common_rows', 0)}

**Value Mismatches**: {len(result.get('value_mismatches', []))} columns have differences
"""
            
            answer += f"\n✅ **Data Identical**: {result['data_identical']}"
            
            return {"messages": [AIMessage(content=answer)]}
        
        elif 'merge' in operation or 'join' in operation:
            # Determine join type from question
            join_type = 'inner'
            if 'left' in question.lower():
                join_type = 'left'
            elif 'right' in question.lower():
                join_type = 'right'
            elif 'outer' in question.lower() or 'full' in question.lower():
                join_type = 'outer'
            elif 'cross' in question.lower():
                join_type = 'cross'
            
            # Try to find common column
            common_cols = list(set(df1.columns) & set(df2.columns))
            on_col = common_cols[0] if common_cols else None
            
            result = csv_comparator.merge_csvs(df1, df2, join_type, on_col)
            
            if not result['success']:
                return {"messages": [AIMessage(content=f"❌ {result.get('error', 'Merge failed')}")]}
            
            answer = f"""🔄 **CSV Merge Complete**

**Join Type**: {join_type.upper()}
**Result**: {result['rows']} rows × {result['columns']} columns
**Join Column**: {on_col if on_col else 'N/A'}

Merged data is ready! You can now query or export it."""
            
            # Update dataframes with merged result
            return {
                "messages": [AIMessage(content=answer)],
                "dataframes": [result['data']],
                "filtered_df": result['data']
            }
        
        elif 'difference' in operation or 'changes' in operation:
            # Find key column
            common_cols = list(set(df1.columns) & set(df2.columns))
            key_col = None
            for col in common_cols:
                if 'id' in col.lower() or 'key' in col.lower():
                    key_col = col
                    break
            
            if not key_col:
                return {"messages": [AIMessage(content="❌ Need a key column (ID) to track changes. Please specify which column to use.")]}
            
            result = csv_comparator.find_differences(df1, df2, key_col, "File 1", "File 2")
            
            if not result['success']:
                return {"messages": [AIMessage(content=f"❌ {result.get('error', 'Difference detection failed')}")]}
            
            summary = result['summary']
            answer = f"""🆚 **Difference Analysis**

**Key Column**: {key_col}

📊 **Summary**:
- ✅ Unchanged: {summary['unchanged_count']} rows
- 🆕 New: {summary['new_count']} rows
- 🗑️ Deleted: {summary['deleted_count']} rows
- ✏️ Modified: {summary['modified_count']} rows

Total changes: {summary['new_count'] + summary['deleted_count'] + summary['modified_count']}"""
            
            # Show examples of modifications
            if result['modified_rows'] and len(result['modified_rows']) > 0:
                answer += "\n\n**Example Modifications**:"
                for mod in result['modified_rows'][:3]:
                    answer += f"\n- {key_col}: {mod[key_col]} → {len(mod['changes'])} fields changed"
            
            return {"messages": [AIMessage(content=answer)]}
        
        elif 'stat' in operation:
            stats1 = csv_comparator.get_statistics(df1)
            stats2 = csv_comparator.get_statistics(df2)
            
            answer = f"""📈 **Statistics Comparison**

**File 1**:
- Shape: {stats1['shape'][0]} rows × {stats1['shape'][1]} columns
- Duplicates: {stats1['duplicates']}
- Missing Values: {sum(stats1['missing_values'].values())} total

**File 2**:
- Shape: {stats2['shape'][0]} rows × {stats2['shape'][1]} columns
- Duplicates: {stats2['duplicates']}
- Missing Values: {sum(stats2['missing_values'].values())} total"""
            
            return {"messages": [AIMessage(content=answer)]}
        
        else:
            # Default to structure comparison
            result = csv_comparator.compare_structure(df1, df2)
            answer = f"Compared structure: {result['structure_identical']}"
            return {"messages": [AIMessage(content=answer)]}
            
    except Exception as e:
        return {"messages": [AIMessage(content=f"❌ Multi-CSV operation failed: {str(e)}")]}
