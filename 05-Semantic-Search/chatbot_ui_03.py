import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Connect to Qdrant
# We assume Qdrant is already running on localhost:6333 from the previous steps
client = QdrantClient(url="http://localhost:6333")
collection_name = "airline_policies_semantic"

# 2. Load Model (Cache it for speed)
# We use the same model as in semantic_search.py
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Airline Policy Chatbot", page_icon="✈️")

st.title("✈️ Airline Policy Chatbot")
st.write("Ask a question about baggage, pets, or flight rules!")

try:
    with st.spinner("Loading AI Model..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
    
# 3. User Input
query = st.text_input("Your Question:", placeholder="Can I bring my cat?")

if query:
    try:
        # 4. Search
        with st.spinner("Searching the knowledge base..."):
            query_vector = model.encode(query).tolist()
            hits = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=3
            ).points

        # 5. Display Results
        st.subheader("Results")
        if not hits:
            st.warning("No results found.")
        
        for hit in hits:
            # Create a card-like display
            with st.container():
                st.markdown(f"**Airline:** {hit.payload['metadata']['airline']} ({hit.payload['metadata']['class']})")
                st.info(hit.payload['text'])
                st.divider()
                
    except Exception as e:
        st.error(f"Error during search: {e}")
