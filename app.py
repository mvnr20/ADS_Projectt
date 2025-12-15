import streamlit as st
import os
import json
import requests
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
from datetime import datetime
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
import re
import time

# ==========================================
# BACKEND LOGIC
# ==========================================

class AIInterviewer:
    def __init__(
        self,
        llm_model: str = "llama3.2:3b",
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        collection_name: str = "job_descriptions"
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        
        # Initialize ChromaDB client
        try:
            self.chroma_client = chromadb.Client(Settings(
                persist_directory="./chroma_interview_db",
                anonymized_telemetry=False
            ))
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Job description embeddings"}
                )
        except Exception as e:
            st.error(f"Database Error: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    page_text = re.sub(r'\s+', ' ', page_text)
                    page_text = re.sub(r'[^\w\s\.,;:!?()-]', '', page_text)
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def intelligent_chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap//4:]) if len(words) > overlap//4 else ""
                current_chunk = overlap_text + " " + paragraph if overlap_text else paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text}
                )
                if response.status_code == 200:
                    embeddings.append(response.json()["embedding"])
                else:
                    embeddings.append([0.0] * 768)
            except:
                embeddings.append([0.0] * 768)
        return embeddings

    def ingest_job_description(self, pdf_path: str, job_id: str) -> str:
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.intelligent_chunk_text(text)
        embeddings = self.generate_embeddings(chunks)
        
        ids = [f"{job_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"job_id": job_id, "chunk_index": i} for i in range(len(chunks))]
        
        self.collection.add(embeddings=embeddings, documents=chunks, ids=ids, metadatas=metadatas)
        return job_id

    def retrieve_context(self, query: str, job_id: str, n_results: int = 5) -> str:
        query_emb = self.generate_embeddings([query])[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where={"job_id": job_id}
        )
        return "\n\n".join(results['documents'][0]) if results['documents'] else ""

    def generate_questions(self, job_id: str, num_questions: int) -> List[Dict]:
        context = self.retrieve_context("requirements skills responsibilities", job_id, 10)
        
        prompt = f"""Generate {num_questions} interview questions based on this job description:
        {context}
        
        Return STRICT JSON array:
        [
          {{
            "question": "text",
            "category": "technical/behavioral/situational",
            "difficulty": "easy/medium/hard",
            "key_skills": ["skill1"]
          }}
        ]"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.7}
                }
            )
            content = response.json()["message"]["content"]
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(content)
        except Exception as e:
            st.error(f"Error generating questions: {e}")
            return []

    def evaluate_answer(self, question: Dict, answer: str, job_id: str) -> Dict:
        context = self.retrieve_context(question['question'], job_id, 3)
        
        prompt = f"""Evaluate this answer based on the job description.
        Job Context: {context}
        Question: {question['question']}
        Answer: {answer}
        
        Return STRICT JSON:
        {{
          "scores": {{
            "technical": <0-10>,
            "communication": <0-10>,
            "relevance": <0-10>
          }},
          "overall_score": <0-10>,
          "feedback": "summary text",
          "improvement": "tips"
        }}"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.2}
                }
            )
            content = response.json()["message"]["content"]
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(content)
        except:
            return {"overall_score": 0, "scores": {}, "feedback": "Error evaluating", "improvement": ""}

# ==========================================
# STREAMLIT INTERFACE
# ==========================================

st.set_page_config(
    page_title="AI Recruiter System", 
    page_icon="🧬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; font-family: 'Segoe UI', sans-serif; }
    .process-card { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #4e8cff; margin-bottom: 20px; }
    .metric-container { background-color: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .metric-value { font-size: 28px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 14px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
    .status-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-right: 10px; }
    .status-ok { background-color: #d4edda; color: #155724; }
    .status-proc { background-color: #cce5ff; color: #004085; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'setup'
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_q_index' not in st.session_state:
    st.session_state.current_q_index = 0
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = []
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'debug_chunks' not in st.session_state:
    st.session_state.debug_chunks = []

# Initialize Backend
@st.cache_resource
def get_interviewer():
    return AIInterviewer()

interviewer = get_interviewer()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧬 System Control")
    st.caption(f"Engine: {interviewer.llm_model}")
    
    st.markdown("### 🛠 Configuration")
    num_questions = st.slider("Interview Length", 3, 10, 5)
    
    st.markdown("### 📊 System Health")
    st.markdown("""
    <div style='margin-bottom:10px'>
        <span class='status-badge status-ok'>Ollama: Connected</span>
        <span class='status-badge status-ok'>ChromaDB: Active</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.job_id:
        st.success(f"Active Job ID: \n{st.session_state.job_id}")

    st.markdown("---")
    # FIX: Updated deprecated use_container_width
    if st.button("🔄 Reset System State", key="reset_btn", help="Clear all data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- PAGE: SETUP ---
if st.session_state.page == 'setup':
    st.markdown("# 🚀 Job Description Setup")
    st.markdown("### Upload a job description to personalize the AI interview process.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("ℹ️ Role requirements will be analyzed and indexed for intelligent questioning")
        # FIX: Added a proper label string "Upload Job Description" to fix the stack trace error
        uploaded_file = st.file_uploader("Upload Job Description", type=['pdf'], label_visibility="collapsed")
    
    with col2:
        if uploaded_file:
            # FIX: Updated deprecated use_container_width
            start_btn = st.button("▶️ Execute Pipeline & Start Interview", type="primary")
            
            if start_btn:
                with st.status("🏗️ Executing RAG Pipeline...", expanded=True) as status:
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    st.write("📄 Extracting raw text from PDF layer...")
                    raw_text = interviewer.extract_text_from_pdf(tmp_path)
                    time.sleep(0.5)
                    
                    st.write("🧩 Applying Intelligent Chunking Strategy...")
                    chunks = interviewer.intelligent_chunk_text(raw_text)
                    st.session_state.debug_chunks = chunks
                    time.sleep(0.5)
                    
                    st.write(f"🔢 Generating {len(chunks)} embeddings...")
                    
                    embeddings = interviewer.generate_embeddings(chunks)
                    ids = [f"{job_id}_chunk_{i}" for i in range(len(chunks))]
                    metadatas = [{"job_id": job_id, "chunk_index": i} for i in range(len(chunks))]
                    interviewer.collection.add(embeddings=embeddings, documents=chunks, ids=ids, metadatas=metadatas)
                    
                    st.write("🧠 Synthesizing Interview Questions via LLM...")
                    questions = interviewer.generate_questions(job_id, num_questions)
                    
                    status.update(label="✅ Pipeline Completed Successfully", state="complete", expanded=False)

                st.markdown("### 🔎 Pipeline Telemetry")
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"<div class='metric-container'><div class='metric-value'>{len(raw_text)}</div><div class='metric-label'>Raw Characters</div></div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-container'><div class='metric-value'>{len(chunks)}</div><div class='metric-label'>Vector Chunks</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-container'><div class='metric-value'>768</div><div class='metric-label'>Embedding Dim</div></div>", unsafe_allow_html=True)

                with st.expander("👁️ Inspect Indexing Data (Engineering Audit)", expanded=True):
                    tab_a, tab_b = st.tabs(["Chunk Distribution", "Sample Vector Payload"])
                    
                    with tab_a:
                        chunk_lens = [len(c) for c in chunks]
                        df_chunks = pd.DataFrame({"Chunk Index": range(len(chunks)), "Length (chars)": chunk_lens})
                        fig = px.bar(df_chunks, x="Chunk Index", y="Length (chars)", title="Chunk Size Uniformity")
                        # FIX: Updated deprecated use_container_width to width="stretch" or similar based on version
                        # Since your logs ask specifically for width="stretch" for container width behavior:
                        st.plotly_chart(fig, width='stretch') 
                    
                    with tab_b:
                        st.json({
                            "chunk_id": f"{job_id}_chunk_0",
                            "content_preview": chunks[0][:200] + "...",
                            "metadata": metadatas[0],
                            "vector_sample": embeddings[0][:5] + ["..."]
                        })

                st.session_state.job_id = job_id
                st.session_state.questions = questions
                st.session_state.page = 'interview'
                os.unlink(tmp_path)
                
                if st.button("Go to Interview Room"):
                    st.rerun()

# --- PAGE: INTERVIEW ---
elif st.session_state.page == 'interview':
    q_index = st.session_state.current_q_index
    total_q = len(st.session_state.questions)
    current_q = st.session_state.questions[q_index]
    
    st.markdown(f"## 🎙️ Technical Interview Session")
    
    col_layout = st.columns([2, 1])
    
    with col_layout[0]:
        st.progress((q_index) / total_q, text=f"Question {q_index + 1} of {total_q}")
        
        st.markdown(f"""
        <div class='process-card'>
            <div style='color: #666; font-size: 14px; margin-bottom: 5px;'>Topic: {current_q.get('category', 'General').upper()} | Difficulty: {current_q.get('difficulty', 'Medium').title()}</div>
            <div style='font-size: 22px; font-weight: 600;'>{current_q['question']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form(key=f"q_form_{q_index}"):
            answer = st.text_area("Candidate Response:", height=250, placeholder="Enter detailed technical response here...")
            
            sub_col1, sub_col2 = st.columns([1, 4])
            with sub_col1:
                submit = st.form_submit_button("Submit Response", type="primary")
            
            if submit and answer:
                with st.spinner("🤖 AI Evaluator is analyzing response against Job Context..."):
                    evaluation = interviewer.evaluate_answer(current_q, answer, st.session_state.job_id)
                    st.session_state.answers.append(answer)
                    st.session_state.evaluations.append(evaluation)
                    
                    if q_index + 1 < total_q:
                        st.session_state.current_q_index += 1
                        st.rerun()
                    else:
                        st.session_state.page = 'results'
                        st.rerun()

    with col_layout[1]:
        st.markdown("### 🛠️ RAG Debugger")
        with st.expander("Retrieved Context", expanded=True):
            st.caption("The AI used these specific chunks from the PDF to generate this question:")
            retrieved = interviewer.retrieve_context(current_q['question'], st.session_state.job_id, n_results=2)
            st.code(retrieved[:600] + "...", language="text")
        
        st.markdown("### 🎯 Expected Key Skills")
        for skill in current_q.get('key_skills', []):
            st.markdown(f"<span class='status-badge status-proc'>{skill}</span>", unsafe_allow_html=True)

# --- PAGE: RESULTS ---
elif st.session_state.page == 'results':
    st.markdown("# 📈 Candidate Evaluation Report")
    
    evals = st.session_state.evaluations
    avg_score = sum(e.get('overall_score', 0) for e in evals) / len(evals) if evals else 0
    
    st.markdown("### Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    rec_color = "#2ecc71" if avg_score > 7 else "#f1c40f" if avg_score > 5 else "#e74c3c"
    rec_text = "STRONG HIRE" if avg_score > 7 else "CONSIDER" if avg_score > 5 else "NO HIRE"
    
    col1.markdown(f"<div class='metric-container' style='border-left: 5px solid {rec_color}'><div class='metric-value'>{avg_score:.1f}/10</div><div class='metric-label'>Overall Fit</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-container'><div class='metric-value'>{len(evals)}</div><div class='metric-label'>Questions Answered</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-container'><div class='metric-value'>{datetime.now().strftime('%b %d')}</div><div class='metric-label'>Interview Date</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-container' style='background-color: {rec_color}; color: white'><div class='metric-value' style='color: white'>{rec_text}</div><div class='metric-label' style='color: white'>Recommendation</div></div>", unsafe_allow_html=True)
    
    st.markdown("---")

    data = []
    for i, e in enumerate(evals):
        scores = e.get('scores', {})
        data.append({
            "Question": f"Q{i+1}",
            "Technical": scores.get('technical', 0),
            "Communication": scores.get('communication', 0),
            "Relevance": scores.get('relevance', 0),
            "Overall": e.get('overall_score', 0)
        })
    df = pd.DataFrame(data)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🕸️ Competency Radar")
        if not df.empty:
            cats = ['Technical', 'Communication', 'Relevance']
            vals = [df[c].mean() for c in cats]
            fig = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill='toself', name='Candidate'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), margin=dict(t=20, b=20))
            # FIX: Updated deprecated use_container_width
            st.plotly_chart(fig, width='stretch')
            
    with c2:
        st.subheader("📊 Performance Trend")
        if not df.empty:
            fig2 = px.line(df, x='Question', y='Overall', markers=True, range_y=[0,10])
            fig2.add_bar(x=df['Question'], y=df['Overall'], opacity=0.3)
            # FIX: Updated deprecated use_container_width
            st.plotly_chart(fig2, width='stretch')

    st.subheader("📝 Question-by-Question Analysis")
    for i, (q, a, e) in enumerate(zip(st.session_state.questions, st.session_state.answers, evals)):
        with st.expander(f"Q{i+1}: {q['question']} (Score: {e.get('overall_score', 0)})"):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown("**Candidate Answer:**")
                st.info(a)
            with col_b:
                st.markdown("**AI Evaluation:**")
                st.write(e.get('feedback'))
                if e.get('improvement'):
                    st.warning(f"💡 Improvement: {e.get('improvement')}")

    st.markdown("### 💾 Data Export")
    full_report = {
        "meta": {"job_id": st.session_state.job_id, "model": interviewer.llm_model},
        "session_data": {"questions": st.session_state.questions, "evaluations": evals}
    }
    st.download_button(
        label="Download Engineering Report (JSON)",
        data=json.dumps(full_report, indent=2),
        file_name=f"interview_report_{st.session_state.job_id}.json",
        mime="application/json"
    )