import streamlit as st
import pandas as pd
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Configuration & Environment Variables ---
# Recommended: Store your API Key in .streamlit/secrets.toml or system environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyDTyilBkx_pN8AbB6fp2IGMWK-UaC8SCGo"

st.set_page_config(page_title="PawPlace: Gen AI Pet-Friendly Finder", page_icon="🐾", layout="wide")

# --- Custom CSS for UI enhancements ---
st.markdown("""
<style>
    .centered-title {
        text-align: center;
        padding-top: 1rem;
    }
    .centered-title h1 {
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .centered-title p {
        font-size: 1.1rem;
        color: #888;
    }
    .big-response {
        font-size: 1.15rem;
        line-height: 1.7;
    }
    .also-like-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #c8a97e;
    }
    .also-like-card h4 {
        margin: 0 0 0.3rem 0;
        color: #2d3436;
    }
    .also-like-card a {
        color: #b08d5b;
        text-decoration: none;
        font-weight: 500;
    }
    .cuisine-tag {
        display: inline-block;
        background: linear-gradient(135deg, #c8a97e, #a67c52);
        color: white;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.3rem 0;
        letter-spacing: 0.3px;
    }
    .welcome-box {
        background: linear-gradient(135deg, #fdf6ec, #f5e6d0);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0 1.5rem 0;
        border: 1px solid #e0cdb5;
    }
    .welcome-box h3 {
        margin: 0 0 0.5rem 0;
        color: #2d3436;
    }
    .welcome-box p {
        color: #636e72;
        margin: 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper: extract cuisine type from Vibe and Name
def extract_cuisine(name, vibe):
    """Infer cuisine type from restaurant name and vibe description."""
    text = f"{name} {vibe}".lower()
    cuisine_map = [
        ("chinese", "🥢 Chinese"), ("cantonese", "🥢 Cantonese"), ("teochew", "🥢 Teochew"),
        ("sichuan", "🌶️ Sichuan"), ("hotpot", "🍲 Hotpot"), ("dim sum", "🥟 Dim Sum"),
        ("italian", "🍝 Italian"), ("french", "🥐 French"), ("spanish", "🥘 Spanish"),
        ("japanese", "🍣 Japanese"), ("korean", "🍜 Korean"), ("indian", "🍛 Indian"),
        ("irani", "☕ Irani Café"), ("persian", "🫓 Persian"),
        ("pan-asian", "🥡 Pan-Asian"), ("asian", "🥡 Asian"),
        ("steakhouse", "🥩 Steakhouse"), ("steak", "🥩 Steakhouse"),
        ("brasserie", "🍽️ Brasserie"), ("pub", "🍺 British Pub"),
        ("scandi", "🌿 Scandi-Japanese"), ("noodle", "🍜 Noodles"),
    ]
    for keyword, label in cuisine_map:
        if keyword in text:
            return label
    return "🍽️ Restaurant"

# --- 2. Data Loading & Vectorization (Core RAG Logic) ---
@st.cache_resource
def initialize_rag():
    # Loading your local CSV database
    try:
        df = pd.read_csv("Pet Restaurant Raw Data.csv")
    except FileNotFoundError:
        st.error("Error: 'Pet Restaurant Raw Data.csv' not found. Please check the file path.")
        return None

    documents = []
    for _, row in df.iterrows():
        # Combine all columns into a rich text block for the AI to retrieve
        content = f"""
        Restaurant Name: {row['Name']}
        Location Area: {row['Location_Area']}
        Vibe: {row['Vibe']}
        Dog Rules: {row['Extracted_Rules']}
        Customer Review: {row['Raw_Pet_Review']}
        """
        # Metadata stores booking/link info for the final display
        cuisine = extract_cuisine(row['Name'], row['Vibe'])
        doc = Document(
            page_content=content, 
            metadata={
                "name": row['Name'], 
                "link": row['Maps_Link'],
                "location": row['Location_Area'],
                "cuisine": cuisine
            }
        )
        documents.append(doc)
    
    # Building the Vector Store (FAISS)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# --- 3. Initialize RAG Engine ---
vectorstore = initialize_rag()

if vectorstore:
    # Set up the Retriever (k=3 means fetching the top 3 most relevant results)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Professional Prompt Template to mitigate Hallucination
    template = """
    You are an expert Pet-Friendly Concierge in London. 
    Use the following pieces of context to answer the user's question.
    If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
    
    Always explain WHY you are recommending a restaurant by referring to its 'Vibe' and 'Dog Rules'.
    Be helpful, professional, and empathetic to dog owners.

    IMPORTANT FORMATTING RULES:
    - Present each restaurant recommendation with its name as a heading (e.g. **Restaurant Name**).
    - After the restaurant name, start a NEW LINE before writing the reasoning (why you recommend it).
    - Use line breaks between different restaurants for readability.

    Context: {context}
    Question: {question}

    Helpful Answer (In English):"""
    
    QA_PROMPT = PromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    # --- 4. Streamlit UI Interface ---
    st.markdown("""
    <div class="centered-title">
        <h1>🐾 PawPlace AI: London's Smart Pet-Friendly Guide</h1>
        <p>Ready to find your pup's next favorite spot?</p>
    </div>
    """, unsafe_allow_html=True)


    # Sidebar: Project Status and Search Suggestions
    with st.sidebar:
        st.header("Project Status")
        st.success(f"Database Loaded: {len(pd.read_csv('Pet Restaurant Raw Data.csv'))} Verified Venues")
        st.markdown("""
        **Try searching for:**
        - "I have a large dog in Canary Wharf."
        - "Romantic vibe in Soho for a date with my puppy."
        - "Sichuan food in Chinatown that allows dogs."
        """)
        st.divider()

    # Chat History Management
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome message on first visit
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-box">
            <h3>👋 Welcome to PawPlace AI!</h3>
            <p>Tell us about your group size, dog size, and what kind of dining experience you're looking for.<br>
            We'll find the perfect pet-friendly spot in London for you.</p>
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        avatar = "🐾" if message["role"] == "assistant" else "🧑"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # User Input Handling
    if prompt := st.chat_input("Tell us what you're looking for (e.g. '2 people with a large dog finding a cozy place for dinner in Soho')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🐾"):
            with st.spinner("Analyzing reviews and pet policies..."):
                # Execute RAG Query
                result = qa_chain.invoke({"query": prompt})
                answer = result["result"]
                source_docs = result["source_documents"]

                # Display AI Response with larger font
                st.markdown(f'<div class="big-response">{answer}</div>', unsafe_allow_html=True)

                # --- 5. "You May Also Like" Section ---
                with st.expander("**🐶 You May Also Like**"):
                    for doc in source_docs:
                        cuisine = doc.metadata.get('cuisine', '🍽️ Restaurant')
                        st.markdown(f"""
                        <div class="also-like-card">
                            <h4>{doc.metadata['name']}</h4>
                            📍 {doc.metadata['location']}<br>
                            <span class="cuisine-tag">{cuisine}</span><br>
                            <a href="{doc.metadata['link']}" target="_blank">View on Google Maps →</a>
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})