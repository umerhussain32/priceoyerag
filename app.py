import streamlit as st
import re
import unicodedata
from retrieval_pipeline import run_streaming_rag

st.set_page_config(page_title="Priceoye Assistant", page_icon="assets/priceoye logo-modified.png",layout="wide")

st.title("PriceOye Assistant", text_alignment="center")
left_space, image_zone, right_space = st.columns([1, 1, 1])

with image_zone:
     
       st.image("assets/priceoye logo-modified.png")
    

with st.sidebar:
    st.title("Details")
    choice = st.write(
    """This is Just prototype. Built Using OpenRouter, Groq, Pincone. It may not provide some information which is present in Database. \n
The **Reason** is that on most models **Groq Api** only provides **8000** tokens limit, exceeding or requesting tokens above this limit may return **ERROR**. And also I have optimized the **prompts** to keep system within token limits. \n
Moreover, this assistant only contains the data of **32 Phones**, prompting it about anything else like **Smart Watches** would return **nothing**."""
),st.divider(), st.sidebar.caption("""Developed by **Umer Hussain** \n
Powered by **Groq, OpenRouter, Pinecone, Firecrawl**. \n
Datasource: **Priceoye** \n
Version: 1.0.01
""")

def is_valid_url(url: str) -> bool:
    return isinstance(url, str) and (url.startswith("http://") or url.startswith("https://"))

def parse_pipe_list(value):
    if not value or not isinstance(value, str):
        return []
    items = [item.strip() for item in value.split("|") if item.strip()]
    return [item for item in items if item.upper() not in ["N/A", "NONE", "NULL", ""]]

# Helper function to normalize unicode spaces (\u202f, \xa0) and punctuation
def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Convert unicode spaces to standard spaces
    text = unicodedata.normalize("NFKD", text)
    # Remove punctuation & extra whitespace for robust string matching
    return re.sub(r"[^\w\s]", "", text).lower().strip()

def render_product_cards(retrieved_docs, ai_text_response=""):
    if not retrieved_docs:
        return

    seen_products = set()
    unique_docs = []
    
    normalized_response = normalize_text(ai_text_response)
    
    for doc in retrieved_docs:
        meta = doc.metadata or {}
        name = meta.get("product_name")
        if not name:
            continue
            
        name_clean = name.strip()
        normalized_name = normalize_text(name_clean)
        
        # Check if normalized product name appears in normalized LLM response
        is_in_text = normalized_name in normalized_response if normalized_response else True
        
        if name_clean not in seen_products and is_in_text:
            seen_products.add(name_clean)
            unique_docs.append(doc)

    if not unique_docs:
        return

    st.divider()
    st.subheader("📦 Product Catalog Cards")
    cols = st.columns(2)

    for idx, doc in enumerate(unique_docs):
        meta = doc.metadata or {}
        
        raw_images = parse_pipe_list(meta.get("color_images"))
        images = [img for img in raw_images if is_valid_url(img)]

        colors = parse_pipe_list(meta.get("color_names"))
        stock_statuses = parse_pipe_list(meta.get("color_stock_status"))
        delivery_badges = parse_pipe_list(meta.get("delivery_badges"))
        box_items = parse_pipe_list(meta.get("whats_in_box") or meta.get("whats_in_box_str"))
        faqs = parse_pipe_list(meta.get("faqs"))
        reviews = parse_pipe_list(meta.get("reviews"))

        with cols[idx % 2]:
            with st.container(border=True):
                if images:
                    st.image(images[0], use_container_width=True)

                st.markdown(f"### {meta.get('product_name', 'Smartphone')}")

                price = meta.get("price_numeric")
                orig_price = meta.get("original_price_numeric")
                discount = meta.get("discount_pct")

                price_str = f"💰 **Price:** Rs. {price:,}" if price else ""
                if orig_price and orig_price > price:
                    price_str += f" ~~(Rs. {orig_price:,})~~"
                if discount:
                    price_str += f" **[{discount}% OFF]**"
                
                if price_str:
                    st.markdown(price_str)
                
                st.write(f"🛡️ **PTA Status:** {'Approved' if meta.get('pta_status') else 'Non-PTA'}")

                if delivery_badges:
                    st.markdown(" • ".join([f"🚀 `{b}`" for b in delivery_badges]))

                if colors:
                    color_info = []
                    for i, color in enumerate(colors):
                        status = stock_statuses[i] if i < len(stock_statuses) else "In Stock"
                        icon = "✅" if "in stock" in status.lower() else "❌"
                        color_info.append(f"{icon} {color.capitalize()}")
                    st.markdown("**Colors Available:** " + ", ".join(color_info))

                if len(images) > 1:
                    with st.expander(f"📷 Image Gallery ({len(images)} photos)"):
                        img_cols = st.columns(min(len(images), 4))
                        for img_idx, img_url in enumerate(images):
                            with img_cols[img_idx % 4]:
                                st.image(img_url, use_container_width=True)

                if box_items:
                    with st.expander("📦 What's in the Box"):
                        for item in box_items:
                            st.write(f"- {item}")

                if faqs:
                    with st.expander(f"❓ FAQs ({len(faqs)})"):
                        for faq in faqs:
                            st.write(f"• {faq}")

                if reviews:
                    with st.expander(f"⭐ Customer Reviews ({len(reviews)})"):
                        for rev in reviews:
                            st.write(f"💬 \"{rev}\"")

                if meta.get("url"):
                    st.link_button("Buy Now 🛒", meta["url"], use_container_width=True)

# ----------------------------------------------------
# UI & STREAMLIT SESSION STATE MEMORY (UPDATED)
# ----------------------------------------------------

# Initialize conversation history tracker
if "messages" not in st.session_state:
    st.session_state.messages = []

# Map avatars cleanly to matching roles
avatars = {"user": None, "assistant": "assets/priceoye logo-modified.png"}

# Render past chat logs correctly without duplicate processing bugs
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=avatars.get(msg["role"])):
        st.markdown(msg["content"])
        if msg.get("retrieved_docs"):
            render_product_cards(msg["retrieved_docs"], msg["content"])

# User chat input element
user_query = st.chat_input("Ask about Phones listed on priceoye.pk...")

if user_query:
    # 1. Render and immediately record user message 
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Render Assistant response container with custom icon
    with st.chat_message("assistant", avatar="assets/priceoye logo-modified.png"):
        # Send full historical dialogue context down to your RAG pipeline
        result = run_streaming_rag(user_query, chat_history=st.session_state.messages[:-1])

        if result is None or result[0] is None:
            msg_text = "We are sorry, but we don't have enough info about that. Try searching something else."
            st.warning(msg_text)
            st.session_state.messages.append({"role": "assistant", "content": msg_text, "retrieved_docs": []})
        else:
            stream_gen, retrieved_docs, analysis = result

            if analysis.get("has_hardcoded_specs") and analysis.get("pinecone_filter"):
                st.info(f"🎯 Filter Applied: `{analysis['pinecone_filter']}`")

            # Stream response generator directly into user screen view
            full_response = st.write_stream(stream_gen)

            # Append context-filtered UI cards below text
            if retrieved_docs:
                render_product_cards(retrieved_docs, full_response)

            # Commit the full interaction to state history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "retrieved_docs": retrieved_docs
            })
