# import os
# import json
# from dotenv import load_dotenv
# from langchain_pinecone import PineconeVectorStore
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI

# load_dotenv()

# # ----------------------------
# # 1. Initialize components
# # ----------------------------
# embeddings = NVIDIAEmbeddings(
#     model="nvidia/nemotron-3-embed-1b:free",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     base_url="https://openrouter.ai/api/v1"
# )

# vectorstore = PineconeVectorStore(
#     index_name="ecommerce-products",
#     embedding=embeddings
# )

# llm = ChatOpenAI(
#     model="openai/gpt-oss-120b",
#     openai_api_key=os.getenv("GROQ_API_KEY"),
#     openai_api_base="https://api.groq.com/openai/v1",
#     temperature=0.0
# )

# # ----------------------------
# # 2. Intelligent Router & Dynamic Query Analyzer
# # ----------------------------
# def analyze_and_route_query(user_query: str, chat_history: list = None) -> dict:
#     """
#     Analyzes the user query:
#     - Identifies casual conversation (greetings, intros) to bypass vector search.
#     - Automatically converts dynamic specs (price, RAM, PTA) into Pinecone metadata filters.
#     - Rewrites and expands queries using chat history context.
#     """
#     history_str = ""
#     if chat_history:
#         formatted_turns = []
#         for msg in chat_history[-4:]:
#             role = "User" if msg["role"] == "user" else "Assistant"
#             formatted_turns.append(f"{role}: {msg['content']}")
#         history_str = "\n".join(formatted_turns)

#     router_prompt = f"""You are a precise query parser for an e-commerce vector database (Pinecone).

# Exact Metadata Schema & Data Types:
# - `price_numeric` (int in PKR): Current selling price (e.g., 30000, 150000). Convert shorthand ("30k" -> 30000, "1.5 lakh" -> 150000).
# - `original_price_numeric` (int in PKR): Original price before discount.
# - `discount_pct` (int): Discount percentage (e.g., 10, 20).
# - `pta_status` (bool): `true` if PTA approved/official, `false` if non-PTA/unapproved.
# - `ram_gb` (int): RAM in GB (e.g., 4, 8, 12, 16).
# - `storage_gb` (int): Storage in GB. Convert units ("1TB" -> 1000 or 1024, "512GB" -> 512).
# - `display_size_inches` (float): Screen size in inches (e.g., 6.5, 6.78).
# - `battery_mah` (int): Battery capacity in mAh (e.g., 5000, 6000).
# - `main_camera_mp` (int): Rear camera resolution in MP (e.g., 50, 108).
# - `color_names` (string): Color name in lowercase (e.g., "black", "blue", "white").
# - `product_name` (string): Model or brand name string.

# Chat History Context (if user asks follow-up questions):
# {history_str if history_str else "No prior history"}

# User Query: "{user_query}"

# Instructions:
# 1. Determine if `is_casual_chat` is true (greetings, intros like "i am umer", questions about identity like "who am i", thank yous, general conversation NOT asking for products).
# 2. If NOT casual chat, extract ALL explicit filter constraints from query and context into `pinecone_filter`.
# 3. Operators to use: `$lte`, `$gte`, `$eq`.
# 4. Range Queries: Combine `$gte` and `$lte` inside the field condition.
# 5. Multiple Conditions: Enclose ALL conditions inside a single top-level `"$and"` array.
# 6. Clean `search_query`: Strip specific numbers already captured in filters.

# Few-Shot Examples:
# - Example 1:
#   Query: "Phones with at least 6.5 inch screen and more than 10% discount under 50k"
#   JSON: {{
#     "is_casual_chat": false,
#     "has_hardcoded_specs": true,
#     "pinecone_filter": {{
#       "$and": [
#         {{"display_size_inches": {{"$gte": 6.5}}}},
#         {{"discount_pct": {{"$gte": 10}}}},
#         {{"price_numeric": {{"$lte": 50000}}}}
#       ]
#     }},
#     "search_query": "large display smartphone on sale"
#   }}

# - Example 2:
#   Query: "hello i am umer"
#   JSON: {{
#     "is_casual_chat": true,
#     "has_hardcoded_specs": false,
#     "pinecone_filter": null,
#     "search_query": ""
#   }}

# Return ONLY a valid JSON object matching this structure:
# {{
#   "is_casual_chat": true/false,
#   "has_hardcoded_specs": true/false,
#   "pinecone_filter": dict or null,
#   "search_query": "string"
# }}

# JSON Output ONLY:"""


#     response = llm.invoke(router_prompt)
#     raw_text = response.content
#     if isinstance(raw_text, list):
#         raw_text = "".join(b.get("text", "") for b in raw_text if isinstance(b, dict) and "text" in b)
    
#     clean_json_str = raw_text.strip().replace("```json", "").replace("```", "").strip()
    
#     try:
#         return json.loads(clean_json_str)
#     except Exception:
#         return {
#             "is_casual_chat": False,
#             "has_hardcoded_specs": False,
#             "pinecone_filter": None,
#             "search_query": user_query
#         }

# # ----------------------------
# # 3. Main RAG Pipeline
# # ----------------------------
# def run_streaming_rag(user_query: str, chat_history: list = None, top_k: int = 5):
#     analysis = analyze_and_route_query(user_query, chat_history)
    
#     # Casual Chat Bypass
#     if analysis.get("is_casual_chat", False):
#         history_str = ""
#         if chat_history:
#             formatted_turns = [f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in chat_history[-4:]]
#             history_str = "\n".join(formatted_turns)
        
#         prompt = f"""You are PriceOye Assistant, an expert mobile sales representative. 
# Respond politely and conversationally to greetings, small talk, or questions about user identity based on the chat history. 
# If the user shares their name or asks who they are, address them by their name explicitly using the provided chat history. 
# Do not list catalog products or execute database responses here. If they ask about completely out-of-scope non-mobile topics, politely pivot back to mobile shopping assistance.

# Chat History:
# {history_str if history_str else "None"}

# Customer Question:
# {user_query}

# Helpful Answer:"""
        
#         def casual_stream():
#             stream = llm.stream(prompt)
#             for chunk in stream:
#                 content = chunk.content
#                 if isinstance(content, str):
#                     yield content
#                 elif isinstance(content, list):
#                     for block in content:
#                         if isinstance(block, dict) and "text" in block:
#                             yield block["text"]
#                         elif isinstance(block, str):
#                             yield block
#         return casual_stream(), [], analysis

#     # --- NEW: Policy Query Detection ---
#     policy_keywords = [
#         "policy", "policies", "installment", "payment", "finance", "emi",
#         "return", "refund", "exchange", "warranty", "guarantee",
#         "privacy", "terms", "conditions", "shipping", "delivery",
#         "cancel", "cancellation", "complaint", "dispute", "legal"
#     ]
#     user_lower = user_query.lower()
#     is_policy = any(kw in user_lower for kw in policy_keywords)

#     if is_policy:
#         # Force search only among policy documents
#         pinecone_filter = {"doc_type": "policy"}
#         # Use the router's search_query if available, else the raw query
#         search_query = analysis.get("search_query", user_query)
#         retrieved_docs = vectorstore.similarity_search(
#             search_query, k=top_k, filter=pinecone_filter
#         )

#         if not retrieved_docs:
#             return None, [], analysis

#         # Build context from policy docs
#         context = "\n\n---\n\n".join(
#             f"Policy Document: {doc.metadata.get('product_name', 'Policy')}\nContent: {doc.page_content}"
#             for doc in retrieved_docs
#         )

#         final_prompt = f"""You are PriceOye Assistant, an E-Commerce Policy Expert.
# Answer the customer's question using ONLY the provided policy documents below.
# If the information is not in the documents, say so politely and offer to connect them with customer support.

# Policy Documents:
# {context}

# Customer Question:
# {user_query}

# Helpful Answer:"""

#         def policy_stream():
#             stream = llm.stream(final_prompt)
#             for chunk in stream:
#                 content = chunk.content
#                 if isinstance(content, str):
#                     yield content
#                 elif isinstance(content, list):
#                     for block in content:
#                         if isinstance(block, dict) and "text" in block:
#                             yield block["text"]
#                         elif isinstance(block, str):
#                             yield block
#         return policy_stream(), retrieved_docs, analysis

#     # Standard Product Vector Search (unchanged)
#     search_query = analysis.get("search_query", user_query)
#     pinecone_filter = analysis.get("pinecone_filter")
    
#     if pinecone_filter:
#         retrieved_docs = vectorstore.similarity_search(search_query, k=top_k, filter=pinecone_filter)
#     else:
#         retrieved_docs = vectorstore.similarity_search(search_query, k=top_k)

#     if not retrieved_docs:
#         return None, [], analysis

#     history_str = ""
#     if chat_history:
#         formatted_turns = [f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in chat_history[-4:]]
#         history_str = "\n".join(formatted_turns)

#     context = "\n\n---\n\n".join(
#         f"Metadata: {doc.metadata}\nContent: {doc.page_content}"
#         for doc in retrieved_docs
#     )

#     final_prompt = f"""You are PriceOye Assistant, an expert E-Commerce Sales & Technical Assistant.
# Answer the customer's question using ONLY the provided product catalog context below.
# Include matching products with their specs, price, PTA status, and box contents. Do not output programming snippets or unrequested creative stories. Focus entirely on product metrics and the user's explicit query context.

# Product Catalog Context:
# {context}

# Recent Chat History:
# {history_str if history_str else "None"}

# Customer Question:
# {user_query}

# Helpful Answer:"""

#     def stream_generator():
#         stream = llm.stream(final_prompt)
#         for chunk in stream:
#             content = chunk.content
#             if isinstance(content, str):
#                 yield content
#             elif isinstance(content, list):
#                 for block in content:
#                     if isinstance(block, dict) and "text" in block:
#                         yield block["text"]
#                     elif isinstance(block, str):
#                         yield block

#     return stream_generator(), retrieved_docs, analysis


import os
import json
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ----------------------------
# 1. Initialize components
# ----------------------------
embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

vectorstore = PineconeVectorStore(
    index_name="ecommerce-products",
    embedding=embeddings
)

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.0
)

# ----------------------------
# 2. Intelligent Router & Dynamic Query Analyzer
# ----------------------------
def analyze_and_route_query(user_query: str, chat_history: list = None) -> dict:
    """
    Analyzes the user query:
    - Identifies casual conversation (greetings, intros) to bypass vector search.
    - Automatically converts dynamic specs (price, RAM, PTA) into Pinecone metadata filters.
    - Rewrites and expands queries using chat history context.
    """
    history_str = ""
    if chat_history:
        formatted_turns = []
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_turns.append(f"{role}: {msg['content']}")
        history_str = "\n".join(formatted_turns)

    router_prompt = f"""You are a precise query parser for an e-commerce vector database (Pinecone).

Exact Metadata Schema & Data Types:
- `price_numeric` (int in PKR): Current selling price (e.g., 30000, 150000). Convert shorthand ("30k" -> 30000, "1.5 lakh" -> 150000).
- `original_price_numeric` (int in PKR): Original price before discount.
- `discount_pct` (int): Discount percentage (e.g., 10, 20).
- `pta_status` (bool): `true` if PTA approved/official, `false` if non-PTA/unapproved.
- `ram_gb` (int): RAM in GB (e.g., 4, 8, 12, 16).
- `storage_gb` (int): Storage in GB. Convert units ("1TB" -> 1000 or 1024, "512GB" -> 512).
- `display_size_inches` (float): Screen size in inches (e.g., 6.5, 6.78).
- `battery_mah` (int): Battery capacity in mAh (e.g., 5000, 6000).
- `main_camera_mp` (int): Rear camera resolution in MP (e.g., 50, 108).
- `color_names` (string): Color name in lowercase (e.g., "black", "blue", "white").
- `product_name` (string): Model or brand name string.

Chat History Context (if user asks follow-up questions):
{history_str if history_str else "No prior history"}

User Query: "{user_query}"

Instructions:
1. Determine if `is_casual_chat` is true (greetings, intros like "i am umer", questions about identity like "who am i", thank yous, general conversation NOT asking for products).
2. If NOT casual chat, extract ALL explicit filter constraints from query and context into `pinecone_filter`.
3. Operators to use: `$lte`, `$gte`, `$gt`, `$lt`, `$eq`.
4. Range Queries: Combine `$gte` and `$lte` inside the field condition.
5. Multiple Conditions: Enclose ALL conditions inside a single top-level `"$and"` array.
6. Clean `search_query`: Strip specific numbers already captured in filters.
7. Product Type RAM Rules:
   - If the user asks for "smartphones", "smart phones", "smartphone", "Android phones", "iPhones", or clearly means a modern smartphone, automatically add:
     {"ram_gb": {"$gt": 1}}
     This means RAM must be greater than 1 GB.
   - If the user asks for "keypad phones", "keypad mobile", "button phones", "feature phones", or clearly means a traditional keypad/button phone, automatically add:
     {"ram_gb": {"$lt": 1}}
     This means RAM must be less than 1 GB.
   - These RAM rules are implicit product-type constraints even when the user does not explicitly mention RAM.
   - Never apply both smartphone and keypad-phone RAM rules to the same query.
   - If the user explicitly specifies a RAM requirement, the user's explicit RAM requirement takes priority over the implicit product-type rule.

Few-Shot Examples:
- Example 1:
  Query: "Phones with at least 6.5 inch screen and more than 10% discount under 50k"
  JSON: {{
    "is_casual_chat": false,
    "has_hardcoded_specs": true,
    "pinecone_filter": {{
      "$and": [
        {{"display_size_inches": {{"$gte": 6.5}}}},
        {{"discount_pct": {{"$gte": 10}}}},
        {{"price_numeric": {{"$lte": 50000}}}}
      ]
    }},
    "search_query": "large display smartphone on sale"
  }}

- Example 2:
  Query: "hello i am umer"
  JSON: {{
    "is_casual_chat": true,
    "has_hardcoded_specs": false,
    "pinecone_filter": null,
    "search_query": ""
  }}
  - Example 3:
  Query: "show me smartphones"
  JSON: {
    "is_casual_chat": false,
    "has_hardcoded_specs": true,
    "pinecone_filter": {
      "$and": [
        {"ram_gb": {"$gt": 1}}
      ]
    },
    "search_query": "smartphone"
  }

- Example 4:
  Query: "show me keypad phones"
  JSON: {
    "is_casual_chat": false,
    "has_hardcoded_specs": true,
    "pinecone_filter": {
      "$and": [
        {"ram_gb": {"$lt": 1}}
      ]
    },
    "search_query": "keypad phone"
  }

Return ONLY a valid JSON object matching this structure:
{{
  "is_casual_chat": true/false,
  "has_hardcoded_specs": true/false,
  "pinecone_filter": dict or null,
  "search_query": "string"
}}

JSON Output ONLY:"""

    response = llm.invoke(router_prompt)
    raw_text = response.content
    if isinstance(raw_text, list):
        raw_text = "".join(b.get("text", "") for b in raw_text if isinstance(b, dict) and "text" in b)
    
    clean_json_str = raw_text.strip().replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(clean_json_str)
    except Exception:
        return {
            "is_casual_chat": False,
            "has_hardcoded_specs": False,
            "pinecone_filter": None,
            "search_query": user_query
        }

# ----------------------------
# 3. Main RAG Pipeline
# ----------------------------
def run_streaming_rag(user_query: str, chat_history: list = None, top_k: int = 5):
    analysis = analyze_and_route_query(user_query, chat_history)
    
    # --- 1. Casual Chat Detection (only if it's pure casual and no policy intent) ---
    is_casual = analysis.get("is_casual_chat", False)
    
    # If casual, but also contains policy keywords, we should NOT bypass – policy takes precedence
    policy_keywords = [
        "policy", "policies", "installment", "payment", "finance", "emi",
        "return", "refund", "exchange", "warranty", "guarantee",
        "privacy", "terms", "conditions", "shipping", "delivery",
        "cancel", "cancellation", "complaint", "dispute", "legal"
    ]
    user_lower = user_query.lower()
    is_policy = any(kw in user_lower for kw in policy_keywords)
    
    # If it's casual but also policy-related, treat as policy (not casual)
    if is_casual and is_policy:
        is_casual = False  # override; policy is more specific
    
    if is_casual:
        # --- Casual Chat (pure greetings, no policy) ---
        history_str = ""
        if chat_history:
            formatted_turns = [f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in chat_history[-4:]]
            history_str = "\n".join(formatted_turns)
        
        prompt = f"""You are PriceOye Assistant, an expert mobile sales representative. 
Respond politely and conversationally to greetings, small talk, or questions about user identity based on the chat history. 
If the user shares their name or asks who they are, address them by their name explicitly using the provided chat history. 
Do not list catalog products or execute database responses here. If they ask about completely out-of-scope non-mobile topics, politely pivot back to mobile shopping assistance.

Chat History:
{history_str if history_str else "None"}

Customer Question:
{user_query}

Helpful Answer:"""
        
        def casual_stream():
            stream = llm.stream(prompt)
            for chunk in stream:
                content = chunk.content
                if isinstance(content, str):
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            yield block["text"]
                        elif isinstance(block, str):
                            yield block
        return casual_stream(), [], analysis

    # --- 2. Policy Query Detection ---
    if is_policy:
        print("🔍 POLICY BRANCH ACTIVATED")
        pinecone_filter = {"doc_type": "policy"}
        search_query = analysis.get("search_query", user_query)
        if not search_query or not search_query.strip():
            search_query = user_query

        retrieved_docs = vectorstore.similarity_search(
            search_query, k=top_k, filter=pinecone_filter
        )

        if not retrieved_docs:
            return None, [], analysis

        # Build context with truncation
        context_parts = []
        total_len = 0
        for doc in retrieved_docs:
            content = doc.page_content
            if total_len + len(content) > 4000:
                break
            context_parts.append(
                f"Policy Document: {doc.metadata.get('product_name', 'Policy')}\nContent: {content}"
            )
            total_len += len(content)
        context = "\n\n---\n\n".join(context_parts)

        system_msg = SystemMessage(
            content="You are a Policy Assistant for PriceOye. "
                    "Answer the customer's question based on the provided policy documents. "
                    "If the question is general (e.g., 'what is your privacy policy'), provide a clear summary of the key points from the documents. "
                    "If the documents contain no relevant information, say 'I don't have that information in our policy documents.' "
                    "Do not use external knowledge."
        )
        human_msg = HumanMessage(
            content=f"Policy Documents:\n{context}\n\nCustomer Question: {user_query}\n\nHelpful Answer:"
        )

        def policy_stream():
            stream = llm.stream([system_msg, human_msg])
            for chunk in stream:
                content = chunk.content
                if isinstance(content, str):
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            yield block["text"]
                        elif isinstance(block, str):
                            yield block

        return policy_stream(), retrieved_docs, analysis

    # --- 3. Standard Product Vector Search (only if not casual and not policy) ---
    search_query = analysis.get("search_query", user_query)
    if not search_query or not search_query.strip():
        search_query = user_query
    
    pinecone_filter = analysis.get("pinecone_filter")
    
    if pinecone_filter:
        retrieved_docs = vectorstore.similarity_search(search_query, k=top_k, filter=pinecone_filter)
    else:
        retrieved_docs = vectorstore.similarity_search(search_query, k=top_k)

    if not retrieved_docs:
        return None, [], analysis

    history_str = ""
    if chat_history:
        formatted_turns = [f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in chat_history[-4:]]
        history_str = "\n".join(formatted_turns)

    context = "\n\n---\n\n".join(
        f"Metadata: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    final_prompt = f"""You are PriceOye Assistant, an expert E-Commerce Sales & Technical Assistant.
Answer the customer's question using ONLY the provided product catalog context below.
Include matching products with their specs, price, PTA status, and box contents. Do not output programming snippets or unrequested creative stories. Focus entirely on product metrics and the user's explicit query context.

Product Catalog Context:
{context}

Recent Chat History:
{history_str if history_str else "None"}

Customer Question:
{user_query}

Helpful Answer:"""

    def stream_generator():
        stream = llm.stream(final_prompt)
        for chunk in stream:
            content = chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        yield block["text"]
                    elif isinstance(block, str):
                        yield block

    return stream_generator(), retrieved_docs, analysis
