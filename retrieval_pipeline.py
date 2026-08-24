import os
import json
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

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
    model="openai/gpt-oss-safeguard-20b",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.05
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
1. Determine if `is_casual_chat` is true (greetings, intros like "i am umer", thank yous, general conversation NOT asking for products).
2. If NOT casual chat, extract ALL explicit filter constraints from query and context into `pinecone_filter`.
3. Operators to use: `$lte`, `$gte`, `$eq`.
4. Range Queries: Combine `$gte` and `$lte` inside the field condition.
5. Multiple Conditions: Enclose ALL conditions inside a single top-level `"$and"` array.
6. Clean `search_query`: Strip specific numbers already captured in filters.

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
def run_streaming_rag(user_query: str, chat_history: list = None, top_k: int = 4):
    analysis = analyze_and_route_query(user_query, chat_history)
    
    # Casual Chat Bypass (Prevents fetching random products on greetings)
    if analysis.get("is_casual_chat", False):
        history_str = ""
        if chat_history:
            formatted_turns = [f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in chat_history[-4:]]
            history_str = "\n".join(formatted_turns)
        
        prompt = f"""You are an e-commerce sales assistant. Respond politely to the customer without listing catalog products.Don't write anything else if user says other than shopping mobiles or not related to pinecone data. JUST POLITELY SAYS "I am just Priceoye Assistant.".

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

    # Standard Product Vector Search
    search_query = analysis.get("search_query", user_query)
    pinecone_filter = analysis.get("pinecone_filter")
    
    if pinecone_filter:
        retrieved_docs = vectorstore.similarity_search(search_query, k=top_k, filter=pinecone_filter)
    else:
        retrieved_docs = vectorstore.similarity_search(search_query, k=top_k)

    if not retrieved_docs:
        return None, [], analysis

    history_str = ""
    if chat_history:
        formatted_turns = [f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in chat_history[-6:]]
        history_str = "\n".join(formatted_turns)

    context = "\n\n---\n\n".join(
        f"Metadata: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    final_prompt = f"""You are an expert E-Commerce Sales & Technical Assistant.
Answer the customer's question using ONLY the provided product catalog context below.
Include matching products with their specs, price, PTA status, and box contents. Don't do anything else like coding, writinging stories etc. I MEAN NOT EVEN SINGLE THING EXCEPT CASUAL CHAT.

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
