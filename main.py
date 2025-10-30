import streamlit as st
import pandas as pd
import requests
import spacy
import pytesseract
import pypdf
import docx
import google.generativeai as genai
from PIL import Image
import io
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(
    layout="wide",
    page_title="MediScan AI - Prescription Analyzer",
    page_icon="💊",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Custom card styling */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: white;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(102,126,234,0.3);
    }
    
    .info-box h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .info-box p {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    /* Medicine table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Upload section */
    .upload-section {
        background: white;
        padding: 3rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: white !important;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Success/Warning/Error messages */
    .stSuccess, .stWarning, .stError {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
    }
    
    /* Feature icons */
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Stats card */
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stat-card h4 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stat-card p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
try:
    genai.configure(api_key="YOUR_API_KEY_HERE")
except Exception as e:
    pass

# Load scispaCy Model
try:
    nlp = spacy.load("en_ner_bc5cdr_md")
except OSError:
    st.error("⚠️ scispaCy 'en_ner_bc5cdr_md' model not found. Please install it first.")
    st.code("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz")
    st.stop()

# --- HELPER FUNCTIONS ---

def extract_text_from_file(uploaded_file):
    """Extracts raw text from various file formats"""
    file_bytes = io.BytesIO(uploaded_file.getvalue())
    file_name = uploaded_file.name
    raw_text = ""
    
    try:
        if file_name.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(file_bytes)
            for page in pdf_reader.pages:
                raw_text += page.extract_text() or ""
                
        elif file_name.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(file_bytes)
            raw_text = pytesseract.image_to_string(image)
            
        elif file_name.endswith(".docx"):
            doc = docx.Document(file_bytes)
            for para in doc.paragraphs:
                raw_text += para.text + "\n"
                
        elif file_name.endswith(".txt"):
            raw_text = file_bytes.read().decode('utf-8')
            
    except Exception as e:
        st.error(f"❌ Error extracting text: {e}")
        return None
        
    return raw_text

def extract_entities(text):
    """Extracts patient name, medicines, and diseases"""
    # Find Patient Name
    patient_name = "Not Found"
    name_match = re.search(r"(?:Patient Name|Patient|Name):\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    if name_match:
        patient_name = name_match.group(1).strip()

    # Find Medicines and Diseases using scispaCy
    doc = nlp(text)
    medicines = set()
    diseases = set()
    
    for ent in doc.ents:
        if ent.label_ == "CHEMICAL":
            medicine_name = ent.text.lower().strip()
            if len(medicine_name) > 3 and medicine_name.replace(" ", "").isalnum():
                medicines.add(medicine_name)
        
        elif ent.label_ == "DISEASE":
            disease_name = ent.text.strip()
            if len(disease_name) > 3:
                diseases.add(disease_name)
                
    return patient_name, list(medicines), list(diseases)

def get_medicine_info(medicine_name):
    """Fetches medicine information from OpenFDA API"""
    base_url = "https://api.fda.gov/drug/label.json"
    search_query = f'openfda.brand_name:"{medicine_name}"+openfda.generic_name:"{medicine_name}"'
    url = f"{base_url}?search=({search_query})&limit=1"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "results" in data:
            purpose = data['results'][0].get('purpose', [])
            if not purpose:
                purpose = data['results'][0].get('indications_and_usage', [])
            
            if purpose:
                return " ".join(purpose)[:400] + "..."
            
    except Exception as e:
        print(f"API Error for {medicine_name}: {e}")
        
    return "No detailed information available from FDA database."

def get_chatbot_response(context, question):
    """RAG-based chatbot using Gemini API"""
    prompt = f"""
    You are MediScan AI, a helpful medical assistant. Answer the user's question based ONLY on the provided prescription context.
    If the answer is not in the context, politely say "I cannot find that information in the uploaded document."

    CONTEXT:
    {context}
    
    USER QUESTION:
    {question}
    
    Provide a clear, concise, and helpful answer.
    """
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "API_KEY_INVALID" in str(e) or "API_KEY" in str(e):
            return "⚠️ Gemini API key is not configured. Please add your API key in the code."
        else:
            return f"❌ Error: {str(e)}"

# --- STREAMLIT APP ---

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_context' not in st.session_state:
    st.session_state.document_context = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Header
st.markdown("""
<div class="main-header">
    <h1>💊 MediScan AI</h1>
    <p>Advanced Prescription Analysis & Medical Assistant</p>
</div>
""", unsafe_allow_html=True)

# Feature highlights
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="feature-icon">📄</div>
        <p>Multi-Format Support</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="feature-icon">🔍</div>
        <p>AI-Powered Analysis</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="feature-icon">💬</div>
        <p>Smart Chatbot</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="feature-icon">⚡</div>
        <p>Instant Results</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# File Upload Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">📤 Upload Prescription</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop your prescription file here",
    type=["pdf", "png", "jpg", "jpeg", "docx", "txt"],
    help="Supported formats: PDF, PNG, JPG, DOCX, TXT"
)

if uploaded_file is not None and not st.session_state.analysis_done:
    with st.spinner('🔄 Analyzing your document... This may take a moment.'):
        raw_text = extract_text_from_file(uploaded_file)
        
        if raw_text:
            # Extract entities
            patient_name, medicine_list, disease_list = extract_entities(raw_text)
            
            st.success("✅ Analysis Complete!")
            
            # Analysis Results Section
            st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Patient Information
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="info-box">
                    <h3>👤 Patient Name</h3>
                    <p>{patient_name}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                diseases_text = ", ".join(disease_list) if disease_list else "None Detected"
                st.markdown(f"""
                <div class="info-box">
                    <h3>🩺 Detected Conditions</h3>
                    <p>{diseases_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Medicine Information
            if medicine_list:
                st.markdown('<h2 class="section-header">💊 Prescribed Medicines</h2>', unsafe_allow_html=True)
                
                table_data = []
                context_pieces = [
                    f"Original document text: {raw_text[:1000]}...",
                    f"Patient Name: {patient_name}",
                    f"Detected Diseases/Conditions: {', '.join(disease_list)}"
                ]
                
                progress_bar = st.progress(0)
                for idx, med in enumerate(medicine_list):
                    info = get_medicine_info(med)
                    table_data.append({
                        "💊 Medicine": med.capitalize(),
                        "📝 Purpose & Usage": info
                    })
                    context_pieces.append(f"Medicine: {med}, Function: {info}")
                    progress_bar.progress((idx + 1) / len(medicine_list))
                
                progress_bar.empty()
                
                # Display medicine table
                df = pd.DataFrame(table_data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=min(400, len(table_data) * 60)
                )
                
            else:
                st.warning("⚠️ No specific medicine names were detected in this document.")
            
            # Store context for chatbot
            st.session_state.document_context = "\n".join(context_pieces)
            st.session_state.messages = []
            st.session_state.analysis_done = True
                
        else:
            st.error("❌ Could not extract any text from the uploaded file.")

st.markdown('</div>', unsafe_allow_html=True)

# Chatbot Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">🤖 Ask MediScan AI</h2>', unsafe_allow_html=True)

if not st.session_state.document_context:
    st.info("💡 Please upload a prescription document to activate the AI assistant.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the prescription (e.g., 'What is this medicine used for?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = get_chatbot_response(
                    context=st.session_state.document_context,
                    question=prompt
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown('</div>', unsafe_allow_html=True)

# Reset button
if st.session_state.analysis_done:
    if st.button("🔄 Analyze New Prescription"):
        st.session_state.analysis_done = False
        st.session_state.document_context = None
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem; margin-top: 2rem;'>
    <p style='font-size: 0.9rem; opacity: 0.8;'>
        MediScan AI | Powered by Gemini & scispaCy | © 2025
    </p>
</div>
""", unsafe_allow_html=True)
