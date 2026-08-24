import streamlit as st
from retrieval_pipeline import run_rag_pipeline

st.set_page_config(page_title="PriceOye AI Assistant", layout="wide")
st.title("📱 PriceOye AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Process user query
if user_query := st.chat_input("Ask about phones (e.g., PTA approved phones under 30000)..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching catalog..."):
            stream, docs = run_rag_pipeline(user_query)

        if not docs:
            msg = "No matching products found."
            st.write(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            # Token Generator for Streamlit Live Streaming
            def generate_tokens():
                for chunk in stream:
                    content = chunk.content
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                yield block["text"]

            full_response = st.write_stream(generate_tokens())
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Render Product Cards
            st.divider()
            st.subheader("🛍️ Matching Products")
            cols = st.columns(3)
            seen_names = set()
            card_idx = 0

            for doc in docs:
                name = doc.metadata.get("product_name")
                if not name or name in seen_names:
                    continue
                seen_names.add(name)

                col = cols[card_idx % 3]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{name}**")
                        price = doc.metadata.get("price_numeric", 0)
                        st.write(f"**Price:** Rs. {price:,}")
                        
                        img_url = doc.metadata.get("color_images")
                        if img_url and str(img_url).startswith("http"):
                            st.image(img_url, use_container_width=True)

                        prod_url = doc.metadata.get("url")
                        if prod_url and str(prod_url).startswith("http"):
                            st.link_button("View Product 🔗", prod_url)

                card_idx += 1