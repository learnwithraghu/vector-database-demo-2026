import streamlit as st
import os
import re
import base64
import lancedb
import pandas as pd
import time

# Ensure we run from the correct directory
DATA_DIR = "share-holder-letters"
DB_URI = "./lancedb_data"
TABLE_NAME = "buffett_letters_multi"

st.set_page_config(page_title="VectorDB Quickstart", layout="wide")

# Centered heading without emoji
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>VectorDB Quickstart: Warren Buffett Letters (2020-2024)</h1>", unsafe_allow_html=True)

# Helper function to display PDF
def display_pdf(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.download_button(
                label="📄 View Full Letter (Download)",
                data=base64.b64decode(base64_pdf),
                file_name=os.path.basename(file_path),
                mime="application/pdf",
                use_container_width=True
            )
        
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error(f"Cannot find PDF file: {file_path}")

# Helper function to simulate processing
def simulate_and_connect_db():
    progress_bar = st.progress(0, text="Starting ingestion process...")
    
    pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
    total_files = len(pdf_files)
    
    if total_files == 0:
        st.error(f"No PDF files found in {DATA_DIR}!")
        return

    # Simulate steps
    for idx, filename in enumerate(pdf_files):
        progress_text = f"Loading PDF {idx+1}/{total_files}: {filename}"
        progress_bar.progress(int((idx / total_files) * 30), text=progress_text)
        time.sleep(0.1) 
            
    progress_bar.progress(40, text="Chunking text into smaller segments...")
    time.sleep(0.3)
            
    progress_bar.progress(60, text="Loading Embedding Model (Sentence-Transformers)...")
    time.sleep(0.4)
    
    progress_bar.progress(80, text="Generating embeddings and saving to VectorDB. This is the magic part! ✨")
    
    try:
        db = lancedb.connect(DB_URI)
        if TABLE_NAME not in db.table_names():
             progress_bar.empty()
             st.error("Preloaded database not found! Please run `python streamlit_app/preload_db.py` first.")
             return
             
        table = db.open_table(TABLE_NAME)
        df_len = len(table.to_pandas())
        
        progress_bar.progress(100, text=f"✅ Successfully vectorized and stored chunks!")
        time.sleep(0.5)
        progress_bar.empty()
        
        st.session_state['db_loaded'] = True
        st.session_state['chunk_count'] = df_len
        st.balloons()
        
    except Exception as e:
        progress_bar.empty()
        st.error(f"Error connecting to VectorDB: {e}")

# Initialize session state
if 'db_loaded' not in st.session_state:
    st.session_state['db_loaded'] = False
if 'chunk_count' not in st.session_state:
    st.session_state['chunk_count'] = 0
if 'active_query' not in st.session_state:
    st.session_state['active_query'] = ""

# Layout Options: Tabs
tab_search, tab_docs = st.tabs(["⚡ Vector Search Engine", "📄 Document Reference"])

with tab_search:
    if not st.session_state['db_loaded']:
        st.info("👋 Welcome! To begin the lab, click the button below to process the documents and initialize the Vector Database.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Store All Letters in VectorDB", type="primary", use_container_width=True):
                simulate_and_connect_db()
                st.rerun()  # Refresh to show metrics and query UI
    else:
        st.success("✅ Database Initialized and Ready!")
        
        # Dashboard Metrics
        st.markdown("### VectorDB Statistics")
        m1, m2, m3 = st.columns(3)
        m1.metric(label="Letters Processed", value="5")
        m2.metric(label="Text Chunks Generated", value=f"{st.session_state['chunk_count']}")
        m3.metric(label="Vector Dimensions", value="384")
        
        st.markdown("---")
        st.markdown("### 🔍 Semantic Search")
        st.write("Type a question to search the vector database. It finds meaning, not just keywords.")
        
        # Search Inputs
        col_input, col_suggestions = st.columns([1, 1])
        
        with col_suggestions:
            st.markdown("**Suggested Queries (Click to Search):**")
            suggestions = [
                "What are Warren Buffett's thoughts on the impact of COVID-19?",
                "How does Berkshire approach stock repurchases and buybacks?",
                "What makes a business a 'good' or 'wonderful' business to own?",
                "What does Charlie Munger say about long-term investing?"
            ]
            
            for sug in suggestions:
                if st.button(sug, use_container_width=True):
                    st.session_state['active_query'] = sug
        
        with col_input:
            user_input = st.text_input("Ask a question about the shareholder letters:", value=st.session_state['active_query'], placeholder="Type your query here and press Enter...")
            
            # If user types a new query, update state
            if user_input != st.session_state['active_query'] and user_input != "":
                st.session_state['active_query'] = user_input
                
        # Execute query if active
        if st.session_state['active_query']:
            query = st.session_state['active_query']
            st.markdown(f"#### Results for: `{query}`")
            
            try:
                db = lancedb.connect(DB_URI)
                table = db.open_table(TABLE_NAME)
                
                with st.spinner("Searching vectors..."):
                    # Since we are showing metrics, let's include distance/similarity if available
                    # LanceDB returns _distance by default in pandas
                    results = table.search(query).limit(3).to_pandas()
                
                if len(results) > 0:
                    for idx, row in results.iterrows():
                        # Format nicely
                        st.markdown(f"""
                        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #ff4b4b;'>
                            <div style='margin-bottom: 10px;'>
                                <span style='background-color: #2e4053; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em;'>📅 Year: {row['year']}</span>
                                <span style='background-color: #5d6d7e; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; margin-left: 10px;'>📄 Source: {row['source']}</span>
                                <span style='background-color: #e67e22; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; float: right; margin-left: 10px;'>🎯 Distance: {row['_distance']:.4f}</span>
                            </div>
                            <div style='font-size: 1.1em; line-height: 1.5;'>
                                "{row['text'].strip()}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No results found.")
            except Exception as e:
                st.error(f"Error querying database: {str(e)}")

with tab_docs:
    st.header("📄 Shareholder Letter Viewer")
    letters = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.pdf')]
    selected_letter = st.selectbox("Select a Letter to view:", letters)
    
    if selected_letter:
        pdf_path = os.path.join(DATA_DIR, selected_letter)
        display_pdf(pdf_path)
