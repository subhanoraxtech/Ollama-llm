"""
Query Rephraser Module
Implements intelligent query reformulation for better retrieval and accuracy
"""

from langchain_ollama import ChatOllama
import pandas as pd
from typing import List, Dict


class QueryRephraser:
    """Reformulates user queries for better retrieval and column matching"""
    
    def __init__(self, llm: ChatOllama):
        self.llm = llm
    
    def rephrase_for_documents(self, query: str, conversation_history: List = None) -> List[str]:
        """
        Generate multiple query variations for document retrieval (RAFT-style)
        
        Args:
            query: Original user query
            conversation_history: Recent conversation context
            
        Returns:
            List of 3 rephrased query variations
        """
        context = ""
        if conversation_history and len(conversation_history) > 0:
            recent = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
            context = f"\n\nRecent conversation:\n" + "\n".join([f"- {msg}" for msg in recent])
        
        prompt = f"""You are a query reformulation expert. Generate 3 different variations of the user's query to improve document retrieval.

Original Query: "{query}"{context}

Generate 3 variations that:
1. Expand abbreviations and add context
2. Use synonyms and related terms
3. Make the query more specific and searchable

Format your response as:
1. [first variation]
2. [second variation]
3. [third variation]

Examples:
Original: "what does it say about payments"
1. payment terms and conditions
2. payment schedule and due dates
3. payment methods and processing information

Original: "next payment due"
1. upcoming payment due date
2. next scheduled payment deadline
3. future payment obligations and dates

Now generate 3 variations for: "{query}"
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the numbered list
            variations = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and clean
                    clean = line.lstrip('0123456789.-) ').strip()
                    if clean:
                        variations.append(clean)
            
            # Return up to 3 variations, or original if parsing failed
            if len(variations) >= 3:
                return variations[:3]
            elif len(variations) > 0:
                return variations + [query] * (3 - len(variations))
            else:
                return [query, query, query]
                
        except Exception as e:
            print(f"Query rephrasing failed: {e}")
            return [query, query, query]
    
    def rephrase_for_csv(self, query: str, df_schema: Dict) -> str:
        """
        Rephrase query for CSV operations with column name mapping
        
        Args:
            query: Original user query
            df_schema: Dictionary with column names and types
            
        Returns:
            Single rephrased query optimized for CSV operations
        """
        columns_info = df_schema.get('columns', [])
        column_names = [col['name'] for col in columns_info] if columns_info else []
        
        prompt = f"""You are a CSV query expert. Rephrase the user's query to be more precise for data retrieval.

User Query: "{query}"

Available Columns: {', '.join(column_names[:20])}

Your task:
1. Map user's informal terms to actual column names
2. Expand abbreviations (e.g., "next payment" → "next payment due date")
3. Make the query more specific about what data to retrieve

Examples:
User: "give me customers with next payment due"
Rephrased: "retrieve customer name, email, and next payment due date for all customers"

User: "show me emails"
Rephrased: "display customer email addresses"

User: "people from california"
Rephrased: "customers where state is California"

Now rephrase: "{query}"

Provide ONLY the rephrased query, no explanation:"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up the response
            rephrased = content.strip().strip('"').strip("'")
            
            # If response is too long or looks like an explanation, return original
            if len(rephrased) > 200 or '\n' in rephrased:
                return query
            
            return rephrased if rephrased else query
            
        except Exception as e:
            print(f"CSV query rephrasing failed: {e}")
            return query
    
    def expand_query(self, query: str) -> List[str]:
        """
        Simple expansion of query with synonyms (fast, no LLM call)
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        q_lower = query.lower()
        expansions = [query]
        
        # Common term mappings
        term_map = {
            'payment': ['payment', 'pay', 'amount due', 'balance'],
            'customer': ['customer', 'client', 'defendant', 'user'],
            'email': ['email', 'e-mail', 'email address', 'contact email'],
            'name': ['name', 'full name', 'customer name'],
            'phone': ['phone', 'telephone', 'phone number', 'contact number'],
            'address': ['address', 'street address', 'location'],
            'date': ['date', 'time', 'when', 'scheduled'],
        }
        
        # Add expansions for matched terms
        for term, synonyms in term_map.items():
            if term in q_lower:
                for syn in synonyms[1:]:  # Skip first (original term)
                    expanded = query.lower().replace(term, syn)
                    if expanded not in [e.lower() for e in expansions]:
                        expansions.append(expanded)
        
        return expansions[:5]  # Limit to 5 variations
    
    def map_user_terms_to_columns(self, query: str, columns: List[str]) -> List[str]:
        """
        Map user's informal terms to actual column names
        
        Args:
            query: User query
            columns: List of actual column names
            
        Returns:
            List of matched column names
        """
        q_lower = query.lower()
        matched = []
        
        # Term to column patterns
        patterns = {
            'payment': ['payment', 'pay', 'due', 'amount', 'balance'],
            'name': ['name', 'first', 'last', 'full'],
            'email': ['email', 'mail', 'e-mail'],
            'phone': ['phone', 'tel', 'mobile', 'cell'],
            'address': ['address', 'street', 'addr', 'location'],
            'state': ['state', 'province', 'region'],
            'city': ['city', 'town'],
            'date': ['date', 'time', 'when', 'due'],
        }
        
        # Check which patterns match the query
        for term, keywords in patterns.items():
            if any(kw in q_lower for kw in keywords):
                # Find columns matching this pattern
                for col in columns:
                    col_lower = col.lower()
                    if any(kw in col_lower for kw in keywords):
                        if col not in matched:
                            matched.append(col)
        
        return matched
