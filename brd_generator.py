import streamlit as st
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
import fitz  # PyMuPDF for better PDF processing
import docx
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image
import base64
from io import BytesIO
import json
import re
import os
import tempfile
from typing import Dict, List, Tuple, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass
import sqlite3
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.schema import HumanMessage, SystemMessage

# Enhanced App Configuration
st.set_page_config(
    page_title="🚀 AI-Powered BRD Generator Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .quality-score {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .compliance-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .badge-excellent { background-color: #10B981; color: white; }
    .badge-good { background-color: #F59E0B; color: white; }
    .badge-needs-attention { background-color: #EF4444; color: white; }
    .timeline-item {
        border-left: 3px solid #667eea;
        padding-left: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Data Models
@dataclass
class BRDSection:
    name: str
    content: Any
    quality_score: float
    compliance_status: str
    last_modified: datetime
    modified_by: str
    comments: List[str]
    
@dataclass
class QualityCheck:
    section: str
    check_type: str
    status: str
    message: str
    severity: str

@dataclass
class User:
    id: str
    name: str
    email: str
    role: str

# Initialize Enhanced Session State
def init_enhanced_session_state():
    defaults = {
        'brd_content': {},
        'extracted_images': {},
        'extracted_formulas': [],
        'generated': False,
        'edited_tables': {},
        'quality_scores': {},
        'compliance_checks': [],
        'document_analysis': {},
        'users': [],
        'current_user': User('user1', 'Current User', 'user@company.com', 'Business Analyst'),
        'comments': {},
        'version_history': [],
        'workflow_status': 'Draft',
        'stakeholders': [],
        'approval_chain': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_enhanced_session_state()

# Enhanced BRD Structure with Quality Metrics
ENHANCED_BRD_STRUCTURE = {
    "1. Executive Summary": {
        "type": "text",
        "description": "High-level summary of business requirements and expected outcomes",
        "quality_criteria": ["completeness", "clarity", "business_value_alignment"],
        "required_elements": ["business_objective", "scope_summary", "success_metrics"]
    },
    "2. Background": {
        "type": "text",
        "description": "Detailed context and background information",
        "quality_criteria": ["regulatory_compliance", "stakeholder_coverage", "historical_context"],
        "required_elements": ["current_state", "drivers_for_change", "regulatory_context"]
    },
    "3. Scope": {
        "type": "parent",
        "subsections": {
            "3.1. In Scope": {
                "type": "table",
                "columns": ["ID", "Description", "Priority", "Owner", "Success Criteria"]
            },
            "3.2. Out of Scope": {
                "type": "table", 
                "columns": ["ID", "Description", "Rationale", "Future Consideration"]
            }
        }
    },
    "4. Stakeholder Analysis": {
        "type": "table",
        "columns": ["Stakeholder", "Role", "Interest Level", "Influence Level", "Communication Strategy", "Approval Required"]
    },
    "5. Assumptions and Dependencies": {
        "type": "parent",
        "subsections": {
            "5.1. Assumptions": {
                "type": "table",
                "columns": ["ID", "Description", "Impact", "Risk Level", "Validation Required"]
            },
            "5.2. Dependencies": {
                "type": "table",
                "columns": ["ID", "Description", "Impact", "Owner", "Target Date", "Status"]
            }
        }
    },
    "6. Business Requirements": {
        "type": "table",
        "columns": ["Unique Rule Ref", "BR ID", "BR Name", "BR Description", "BR Owner", "BR Type", "Priority", "Success Criteria", "Acceptance Criteria"]
    },
    "7. Functional Requirements": {
        "type": "table",
        "columns": ["FR ID", "FR Name", "Description", "Related BR", "Priority", "Complexity", "Owner", "Status"]
    },
    "8. Non-Functional Requirements": {
        "type": "table",
        "columns": ["NFR ID", "Category", "Description", "Metric", "Target Value", "Priority"]
    },
    "9. Risk Assessment": {
        "type": "table",
        "columns": ["Risk ID", "Description", "Probability", "Impact", "Risk Score", "Mitigation Strategy", "Owner"]
    },
    "10. Applicable Regulations": {
        "type": "table",
        "columns": ["Unique Rule Ref", "Regulation", "Section", "Regulatory Text", "Compliance Requirement", "Impact Assessment"]
    },
    "11. Implementation Timeline": {
        "type": "table",
        "columns": ["Phase", "Milestone", "Description", "Start Date", "End Date", "Dependencies", "Owner"]
    },
    "12. Success Metrics and KPIs": {
        "type": "table",
        "columns": ["Metric ID", "Metric Name", "Description", "Baseline", "Target", "Measurement Method", "Frequency"]
    },
    "13. Approval Matrix": {
        "type": "table",
        "columns": ["Role", "Name", "Responsibility", "Approval Level", "Date Required", "Status"]
    },
    "14. Appendix": {
        "type": "table",
        "columns": ["ID", "Name", "Description", "Type", "Location"]
    }
}

# AI-Powered Document Analysis
def analyze_document_intelligence(text: str, images: dict, formulas: list) -> dict:
    """Advanced AI-powered document analysis"""
    analysis = {
        'document_type': 'Unknown',
        'regulatory_framework': [],
        'key_entities': [],
        'complexity_score': 0,
        'compliance_indicators': [],
        'stakeholder_mentions': [],
        'risk_indicators': [],
        'timeline_references': []
    }
    
    # Document type detection
    doc_type_indicators = {
        'regulatory': ['regulation', 'compliance', 'requirement', 'shall', 'must'],
        'policy': ['policy', 'procedure', 'guideline', 'standard'],
        'technical': ['specification', 'technical', 'architecture', 'design'],
        'business': ['business', 'process', 'workflow', 'operation']
    }
    
    text_lower = text.lower()
    for doc_type, indicators in doc_type_indicators.items():
        score = sum(1 for indicator in indicators if indicator in text_lower)
        if score > 0:
            analysis['document_type'] = doc_type.title()
            break
    
    # Regulatory framework detection
    frameworks = ['sox', 'gdpr', 'basel', 'mifid', 'dodd-frank', 'pci-dss', 'iso 27001']
    analysis['regulatory_framework'] = [fw for fw in frameworks if fw in text_lower]
    
    # Key entity extraction (simplified)
    entity_patterns = [
        r'\b[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+\b',  # Names
        r'\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b',    # Dates
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',     # Money
    ]
    
    for pattern in entity_patterns:
        matches = re.findall(pattern, text)
        analysis['key_entities'].extend(matches[:10])  # Limit results
    
    # Complexity scoring
    complexity_factors = [
        len(text) > 50000,  # Large document
        len(images) > 10,   # Many images
        len(formulas) > 5,  # Complex formulas
        len(analysis['regulatory_framework']) > 2,  # Multiple regulations
    ]
    analysis['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
    
    return analysis

# Enhanced Quality Assessment Engine
def calculate_quality_score(section_name: str, content: Any, structure_config: dict) -> Tuple[float, List[QualityCheck]]:
    """Calculate quality score and generate quality checks"""
    checks = []
    score = 0.0
    max_score = 100.0
    
    # Basic completeness check
    if content and str(content).strip():
        score += 30
        checks.append(QualityCheck(section_name, "completeness", "PASS", "Section has content", "info"))
    else:
        checks.append(QualityCheck(section_name, "completeness", "FAIL", "Section is empty", "error"))
    
    # Structure-specific checks
    if structure_config.get("type") == "table":
        if isinstance(content, pd.DataFrame) and not content.empty:
            score += 30
            checks.append(QualityCheck(section_name, "format", "PASS", "Proper table format", "info"))
            
            # Check for minimum rows
            if len(content) >= 3:
                score += 20
                checks.append(QualityCheck(section_name, "content_depth", "PASS", "Sufficient detail provided", "info"))
            else:
                checks.append(QualityCheck(section_name, "content_depth", "WARNING", "Consider adding more detail", "warning"))
                
            # Check for required columns
            required_cols = structure_config.get("columns", [])
            if all(col in content.columns for col in required_cols):
                score += 20
                checks.append(QualityCheck(section_name, "column_compliance", "PASS", "All required columns present", "info"))
        else:
            checks.append(QualityCheck(section_name, "format", "FAIL", "Should be in table format", "error"))
    
    elif structure_config.get("type") == "text":
        if isinstance(content, str) and len(content) > 100:
            score += 40
            checks.append(QualityCheck(section_name, "detail_level", "PASS", "Adequate detail provided", "info"))
            
            # Check for required elements
            required_elements = structure_config.get("required_elements", [])
            elements_found = 0
            for element in required_elements:
                if element.replace("_", " ").lower() in content.lower():
                    elements_found += 1
            
            if required_elements:
                element_score = (elements_found / len(required_elements)) * 30
                score += element_score
                if elements_found == len(required_elements):
                    checks.append(QualityCheck(section_name, "required_elements", "PASS", "All required elements present", "info"))
                else:
                    checks.append(QualityCheck(section_name, "required_elements", "WARNING", f"Missing {len(required_elements) - elements_found} required elements", "warning"))
        else:
            checks.append(QualityCheck(section_name, "detail_level", "WARNING", "Consider adding more detail", "warning"))
    
    return min(score, max_score), checks

# Advanced Visualization Components
def create_compliance_dashboard():
    """Create an interactive compliance dashboard"""
    st.subheader("🎯 Compliance Dashboard")
    
    # Calculate overall metrics
    total_sections = len(ENHANCED_BRD_STRUCTURE)
    completed_sections = len(st.session_state.brd_content)
    
    # Quality scoring
    quality_scores = []
    for section_name, content in st.session_state.brd_content.items():
        if section_name in ENHANCED_BRD_STRUCTURE:
            score, _ = calculate_quality_score(section_name, content, ENHANCED_BRD_STRUCTURE[section_name])
            quality_scores.append(score)
    
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="quality-score">{:.0f}%</div>
            <div>Overall Quality</div>
        </div>
        """.format(avg_quality), unsafe_allow_html=True)
    
    with col2:
        completion_rate = (completed_sections / total_sections) * 100
        st.markdown("""
        <div class="metric-card">
            <div class="quality-score">{:.0f}%</div>
            <div>Completion Rate</div>
        </div>
        """.format(completion_rate), unsafe_allow_html=True)
    
    with col3:
        risk_count = len([c for c in st.session_state.compliance_checks if c.severity == 'error'])
        st.markdown("""
        <div class="metric-card">
            <div class="quality-score">{}</div>
            <div>High Risk Items</div>
        </div>
        """.format(risk_count), unsafe_allow_html=True)
    
    with col4:
        pending_approvals = len([a for a in st.session_state.approval_chain if a.get('status') == 'pending'])
        st.markdown("""
        <div class="metric-card">
            <div class="quality-score">{}</div>
            <div>Pending Approvals</div>
        </div>
        """.format(pending_approvals), unsafe_allow_html=True)
    
    # Quality trend chart
    if quality_scores:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(quality_scores) + 1)),
            y=quality_scores,
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#667eea', width=3)
        ))
        fig.update_layout(
            title="Quality Score by Section",
            xaxis_title="Section Number",
            yaxis_title="Quality Score (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

def create_stakeholder_matrix():
    """Create interactive stakeholder influence/interest matrix"""
    st.subheader("👥 Stakeholder Analysis Matrix")
    
    # Sample stakeholder data (in real app, this would come from the BRD)
    stakeholders_data = [
        {"name": "Business Sponsor", "interest": 9, "influence": 9, "category": "Champion"},
        {"name": "Compliance Officer", "interest": 8, "influence": 7, "category": "Key Player"},
        {"name": "IT Team", "interest": 6, "influence": 8, "category": "Key Player"},
        {"name": "End Users", "interest": 7, "influence": 4, "category": "Subject"},
        {"name": "Legal Team", "interest": 8, "influence": 6, "category": "Key Player"},
    ]
    
    df_stakeholders = pd.DataFrame(stakeholders_data)
    
    fig = px.scatter(
        df_stakeholders, 
        x="interest", 
        y="influence",
        text="name",
        color="category",
        size_max=60,
        title="Stakeholder Interest vs Influence Matrix"
    )
    
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Interest Level (1-10)",
        yaxis_title="Influence Level (1-10)",
        height=400
    )
    
    # Add quadrant lines
    fig.add_hline(y=5, line_dash="dash", line_color="gray")
    fig.add_vline(x=5, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)

def create_risk_heatmap():
    """Create risk assessment heatmap"""
    st.subheader("🔥 Risk Heat Map")
    
    # Sample risk data
    risks = [
        {"risk": "Regulatory Changes", "probability": 7, "impact": 9},
        {"risk": "Timeline Delays", "probability": 6, "impact": 6},
        {"risk": "Budget Overrun", "probability": 5, "impact": 7},
        {"risk": "Stakeholder Conflicts", "probability": 4, "impact": 5},
        {"risk": "Technical Complexity", "probability": 8, "impact": 6},
    ]
    
    df_risks = pd.DataFrame(risks)
    df_risks["risk_score"] = df_risks["probability"] * df_risks["impact"]
    
    fig = px.scatter(
        df_risks,
        x="probability",
        y="impact",
        size="risk_score",
        text="risk",
        color="risk_score",
        color_continuous_scale="Reds",
        title="Risk Probability vs Impact Analysis"
    )
    
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Probability (1-10)",
        yaxis_title="Impact (1-10)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Enhanced LLM Configuration with Intelligence
@st.cache_resource
def init_enhanced_llm():
    """Initialize ChatOpenAI with custom configuration"""
    return ChatOpenAI(
        base_url="http://lwnde002xdgpu.sdi.corp.bankofamerica.com:8123/v1",
        api_key="dummy",
        model="/phoenix/workspaces/nbkm74lv/llama3.3-4bit-awq",
        temperature=0.3,
        max_tokens=4000,
        streaming=False
    )

def generate_intelligent_brd_section(llm: ChatOpenAI, section_name, section_config, document_text, images, formulas, document_analysis):
    """Generate BRD section with enhanced AI intelligence using ChatOpenAI"""
    
    # Context enhancement based on document analysis
    context_enhancement = f"""
    Document Analysis Context:
    - Document Type: {document_analysis.get('document_type', 'Unknown')}
    - Regulatory Frameworks: {', '.join(document_analysis.get('regulatory_framework', []))}
    - Complexity Level: {document_analysis.get('complexity_score', 0):.1f}/1.0
    - Key Entities: {', '.join(document_analysis.get('key_entities', [])[:5])}
    """
    
    media_context = ""
    if images:
        media_context += f"\nAvailable Images: {', '.join(images.keys())}\n"
    if formulas:
        media_context += f"\nExtracted Formulas:\n" + "\n".join(formulas[:10])
    
    # Enhanced prompts based on section type
    if section_config["type"] == "table":
        columns = section_config["columns"]
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are an expert business analyst creating a high-quality "{section_name}" section for a Business Requirements Document.
        
        {context_enhancement}
        
        Create a comprehensive table with exactly these columns: {' | '.join(columns)}
        
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Based on this regulatory text: {document_text[:3000]}
        {media_context}
        
        Requirements:
        1. Generate 5-8 detailed, realistic rows
        2. Ensure each entry is specific and actionable
        3. Use proper business terminology
        4. Reference images using [IMAGE: image_key] format where relevant
        5. Include risk assessments and priorities where applicable
        
        Return in pipe-separated format:
        {' | '.join(columns)}
        Row1Value1 | Row1Value2 | Row1Value3...
        Row2Value1 | Row2Value2 | Row2Value3...
        """
    else:
        description = section_config.get("description", f"Generate content for {section_name}")
        required_elements = section_config.get("required_elements", [])
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are an expert business analyst creating a high-quality "{section_name}" section for a Business Requirements Document.
        
        {context_enhancement}
        
        Section Purpose: {description}
        
        Required Elements to Include: {', '.join(required_elements)}
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Based on this regulatory text: {document_text[:3500]}
        {media_context}
        
        Requirements:
        1. Provide comprehensive, professional content (minimum 300 words)
        2. Address all required elements explicitly
        3. Use clear, business-appropriate language
        4. Include specific examples and metrics where applicable
        5. Reference images using [IMAGE: image_key] format where relevant
        6. Structure with appropriate headings and bullet points
        """
    
    try:
        # Create message objects for ChatOpenAI
        system_message = SystemMessage(
            content="You are an expert business analyst with deep knowledge of regulatory compliance, business process optimization, and stakeholder management. Create professional, detailed, and actionable BRD content."
        )
        human_message = HumanMessage(content=user_prompt)
        
        # Get response from ChatOpenAI
        response = llm([system_message, human_message])
        return response.content
    except Exception as e:
        st.error(f"Error generating {section_name}: {str(e)}")
        return f"Error generating {section_name} section"

# Enhanced UI Components
def render_enhanced_header():
    """Render enhanced header with branding"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI-Powered BRD Generator Pro</h1>
        <p>Transform regulatory documents into comprehensive Business Requirements Documents with advanced AI intelligence, real-time collaboration, and compliance tracking</p>
    </div>
    """, unsafe_allow_html=True)

def render_workflow_timeline():
    """Render workflow timeline"""
    st.subheader("📋 Workflow Timeline")
    
    timeline_steps = [
        {"step": "Document Upload", "status": "completed", "date": "Today"},
        {"step": "AI Analysis", "status": "completed", "date": "Today"},
        {"step": "BRD Generation", "status": "in_progress", "date": "Today"},
        {"step": "Quality Review", "status": "pending", "date": "Tomorrow"},
        {"step": "Stakeholder Approval", "status": "pending", "date": "Next Week"},
        {"step": "Final Sign-off", "status": "pending", "date": "TBD"}
    ]
    
    for step in timeline_steps:
        status_color = {
            "completed": "#10B981",
            "in_progress": "#F59E0B", 
            "pending": "#6B7280"
        }[step["status"]]
        
        status_icon = {
            "completed": "✅",
            "in_progress": "⏳",
            "pending": "⭕"
        }[step["status"]]
        
        st.markdown(f"""
        <div class="timeline-item" style="border-left-color: {status_color};">
            {status_icon} <strong>{step['step']}</strong> - {step['date']}<br>
            <small style="color: {status_color};">{step['status'].replace('_', ' ').title()}</small>
        </div>
        """, unsafe_allow_html=True)

# Main Application
def main():
    render_enhanced_header()
    
    # Sidebar with enhanced options
    st.sidebar.title("🚀 BRD Generator Pro")
    st.sidebar.markdown("Advanced AI-powered document transformation")
    
    # User profile section
    st.sidebar.subheader("👤 User Profile")
    user = st.session_state.current_user
    st.sidebar.info(f"**{user.name}**\n{user.role}\n📧 {user.email}")
    
    # File upload section
    st.sidebar.subheader("📁 Document Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Regulatory Document",
        type=['pdf', 'docx', 'txt'],
        help="Support for documents up to 500 pages with advanced AI analysis"
    )
    
    # Enhanced extraction options
    st.sidebar.subheader("🎨 AI Enhancement Options")
    extract_images = st.sidebar.checkbox("Extract & Analyze Images", value=True)
    extract_formulas = st.sidebar.checkbox("Detect Mathematical Formulas", value=True)
    intelligent_analysis = st.sidebar.checkbox("Advanced Document Intelligence", value=True)
    stakeholder_detection = st.sidebar.checkbox("Auto-detect Stakeholders", value=True)
    risk_assessment = st.sidebar.checkbox("AI Risk Assessment", value=True)
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        max_images = st.slider("Max Images to Extract", 1, 100, 30)
        quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, 0.7)
        collaboration_mode = st.checkbox("Enable Real-time Collaboration", value=False)
        
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Analysis", "📋 BRD Generation", "📊 Analytics", "👥 Collaboration"])
    
    with tab1:
        if uploaded_file is not None:
            # Document processing and analysis
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.success(f"📄 **{uploaded_file.name}** uploaded successfully ({file_size_mb:.2f} MB)")
            
            # Enhanced document extraction
            with st.spinner("🧠 Performing advanced AI analysis..."):
                if uploaded_file.type == "application/pdf":
                    document_text, extracted_images, extracted_formulas = extract_images_and_formulas_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    document_text, extracted_images = extract_images_from_docx(uploaded_file)
                    extracted_formulas = []
                else:
                    uploaded_file.seek(0)
                    document_text = str(uploaded_file.read(), "utf-8")
                    extracted_images = {}
                    extracted_formulas = []
                
                # Store in session state
                st.session_state.extracted_images = extracted_images if extract_images else {}
                st.session_state.extracted_formulas = extracted_formulas if extract_formulas else []
                
                # Perform intelligent analysis
                if intelligent_analysis and document_text:
                    st.session_state.document_analysis = analyze_document_intelligence(
                        document_text, extracted_images, extracted_formulas
                    )
            
            # Enhanced metrics display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📝 Content", f"{len(document_text):,} chars")
            with col2:
                st.metric("🖼️ Images", len(st.session_state.extracted_images))
            with col3:
                st.metric("🧮 Formulas", len(st.session_state.extracted_formulas))
            with col4:
                complexity = st.session_state.document_analysis.get('complexity_score', 0)
                st.metric("🎯 Complexity", f"{complexity:.1f}")
            
            # Document intelligence insights
            if st.session_state.document_analysis:
                st.subheader("🧠 AI Document Analysis")
                analysis = st.session_state.document_analysis
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Document Type:** {analysis.get('document_type', 'Unknown')}")
                    if analysis.get('regulatory_framework'):
                        frameworks = ', '.join(analysis['regulatory_framework'])
                        st.info(f"**Regulatory Frameworks:** {frameworks}")
                
                with col2:
                    if analysis.get('key_entities'):
                        st.success(f"**Key Entities Found:** {len(analysis['key_entities'])}")
                    if analysis.get('stakeholder_mentions'):
                        st.success(f"**Stakeholders Detected:** {len(analysis['stakeholder_mentions'])}")
                
                # Preview extracted content
                with st.expander("🔍 Content Preview", expanded=False):
                    preview_text = document_text[:2000] + "..." if len(document_text) > 2000 else document_text
                    st.text_area("Document Content", preview_text, height=200, disabled=True)
                
                # Media preview with enhanced display
                if st.session_state.extracted_images or st.session_state.extracted_formulas:
                    with st.expander("🎨 Extracted Media Gallery", expanded=False):
                        if st.session_state.extracted_images:
                            st.subheader("📸 Images")
                            cols = st.columns(4)
                            for idx, (img_key, img_b64) in enumerate(st.session_state.extracted_images.items()):
                                with cols[idx % 4]:
                                    display_image_from_base64(img_b64, caption=img_key, max_width=150)
                        
                        if st.session_state.extracted_formulas:
                            st.subheader("📐 Mathematical Formulas")
                            for i, formula in enumerate(st.session_state.extracted_formulas[:15]):
                                with st.expander(f"Formula {i+1}"):
                                    st.code(formula, language="text")
        else:
            st.info("👆 Please upload a document to begin AI-powered analysis")
            
            # Sample document showcase
            st.subheader("📚 Sample Documents")
            sample_docs = [
                {"name": "GDPR Compliance Guide", "type": "Regulatory", "pages": 89, "complexity": "High"},
                {"name": "SOX Internal Controls", "type": "Financial", "pages": 156, "complexity": "Medium"},
                {"name": "API Security Standards", "type": "Technical", "pages": 45, "complexity": "Medium"},
            ]
            
            for doc in sample_docs:
                with st.expander(f"📄 {doc['name']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Type:** {doc['type']}")
                    with col2:
                        st.write(f"**Pages:** {doc['pages']}")
                    with col3:
                        st.write(f"**Complexity:** {doc['complexity']}")
    
    with tab2:
        if uploaded_file is not None and 'document_text' in locals():
            # Enhanced BRD generation
            st.subheader("🚀 AI-Powered BRD Generation")
            
            # Generation options
            col1, col2, col3 = st.columns(3)
            with col1:
                template_type = st.selectbox(
                    "📋 BRD Template",
                    ["Standard Enterprise", "Regulatory Compliance", "Technical Integration", "Business Process"]
                )
            with col2:
                quality_level = st.selectbox(
                    "🎯 Quality Level",
                    ["Standard", "Premium", "Enterprise"]
                )
            with col3:
                stakeholder_focus = st.selectbox(
                    "👥 Stakeholder Focus",
                    ["Balanced", "Business-Heavy", "Technical-Heavy", "Compliance-Heavy"]
                )
            
            # Advanced generation options
            with st.expander("⚙️ Advanced Generation Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    include_risk_analysis = st.checkbox("Include Risk Analysis", value=True)
                    include_timeline = st.checkbox("Include Implementation Timeline", value=True)
                    include_kpis = st.checkbox("Include Success Metrics", value=True)
                with col2:
                    auto_stakeholder_mapping = st.checkbox("Auto-map Stakeholders", value=True)
                    compliance_validation = st.checkbox("Compliance Validation", value=True)
                    generate_appendices = st.checkbox("Generate Appendices", value=True)
            
            # Generate button with enhanced styling
            if st.button("🚀 Generate Enhanced BRD", type="primary", use_container_width=True):
                llm = init_enhanced_llm()
                
                # Enhanced progress tracking
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stage_info = st.empty()
                
                total_sections = len(ENHANCED_BRD_STRUCTURE)
                section_count = 0
                
                # Pre-generation analysis
                status_text.text("🧠 Performing pre-generation analysis...")
                stage_info.info("Analyzing document structure and complexity...")
                
                for section_name, section_config in ENHANCED_BRD_STRUCTURE.items():
                    if section_config.get("type") == "parent":
                        st.session_state.brd_content[section_name] = {}
                        for subsection_name, subsection_config in section_config["subsections"].items():
                            status_text.text(f"🔄 Generating {subsection_name}...")
                            stage_info.info(f"Using AI intelligence for: {subsection_name}")
                            
                            content = generate_intelligent_brd_section(
                                llm, subsection_name, subsection_config, document_text,
                                st.session_state.extracted_images, 
                                st.session_state.extracted_formulas,
                                st.session_state.document_analysis
                            )
                            
                            if subsection_config["type"] == "table":
                                df = parse_table_content(content, subsection_config["columns"])
                                st.session_state.brd_content[section_name][subsection_name] = df
                            else:
                                st.session_state.brd_content[section_name][subsection_name] = content
                            
                            section_count += 1
                            progress_bar.progress(section_count / total_sections)
                    else:
                        status_text.text(f"🔄 Generating {section_name}...")
                        stage_info.info(f"Applying quality criteria for: {section_name}")
                        
                        content = generate_intelligent_brd_section(
                            llm, section_name, section_config, document_text,
                            st.session_state.extracted_images, 
                            st.session_state.extracted_formulas,
                            st.session_state.document_analysis
                        )
                        
                        if section_config["type"] == "table":
                            df = parse_table_content(content, section_config["columns"])
                            st.session_state.brd_content[section_name] = df
                        else:
                            st.session_state.brd_content[section_name] = content
                        
                        section_count += 1
                        progress_bar.progress(section_count / total_sections)
                
                # Post-generation quality analysis
                status_text.text("✨ Performing quality analysis...")
                stage_info.info("Running AI-powered quality checks...")
                
                # Calculate quality scores
                st.session_state.compliance_checks = []
                st.session_state.quality_scores = {}
                
                for section_name, content in st.session_state.brd_content.items():
                    if section_name in ENHANCED_BRD_STRUCTURE:
                        score, checks = calculate_quality_score(
                            section_name, content, ENHANCED_BRD_STRUCTURE[section_name]
                        )
                        st.session_state.quality_scores[section_name] = score
                        st.session_state.compliance_checks.extend(checks)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Enhanced BRD Generation Complete!")
                stage_info.success("🎉 Your professional BRD is ready for review!")
                st.session_state.generated = True
                st.balloons()
            
            # Display generated content with enhanced editing
            if st.session_state.generated and st.session_state.brd_content:
                st.markdown("---")
                st.header("📋 Enhanced BRD - Review & Edit")
                
                # Quality overview
                if st.session_state.quality_scores:
                    avg_quality = sum(st.session_state.quality_scores.values()) / len(st.session_state.quality_scores)
                    quality_color = "🟢" if avg_quality >= 80 else "🟡" if avg_quality >= 60 else "🔴"
                    st.success(f"{quality_color} **Overall Quality Score: {avg_quality:.1f}%**")
                
                # Section tabs for editing
                section_tabs = st.tabs([name.split('.')[0] + "." for name in st.session_state.brd_content.keys()])
                
                for i, (section_name, content) in enumerate(st.session_state.brd_content.items()):
                    with section_tabs[i]:
                        # Section header with quality indicator
                        quality_score = st.session_state.quality_scores.get(section_name, 0)
                        quality_badge = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.subheader(section_name)
                        with col2:
                            st.metric("Quality", f"{quality_score:.0f}%")
                        with col3:
                            st.write(f"Status {quality_badge}")
                        
                        # Section-specific quality checks
                        section_checks = [c for c in st.session_state.compliance_checks if c.section == section_name]
                        if section_checks:
                            with st.expander("🔍 Quality Insights"):
                                for check in section_checks:
                                    icon = "✅" if check.status == "PASS" else "⚠️" if check.status == "WARNING" else "❌"
                                    st.write(f"{icon} **{check.check_type.title()}:** {check.message}")
                        
                        # Content editing
                        if isinstance(content, dict):
                            for subsection_name, subcontent in content.items():
                                st.write(f"**{subsection_name}**")
                                
                                if isinstance(subcontent, pd.DataFrame):
                                    # Enhanced table editor
                                    st.write("📊 Interactive Table Editor:")
                                    edited_df = st.data_editor(
                                        subcontent,
                                        use_container_width=True,
                                        num_rows="dynamic",
                                        column_config={
                                            col: st.column_config.TextColumn(
                                                help=f"Edit {col} values"
                                            ) for col in subcontent.columns
                                        },
                                        key=f"enhanced_table_{section_name}_{subsection_name}"
                                    )
                                    st.session_state.brd_content[section_name][subsection_name] = edited_df
                                    
                                    # Table analytics
                                    if not edited_df.empty:
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Rows", len(edited_df))
                                        with col2:
                                            st.metric("Columns", len(edited_df.columns))
                                        with col3:
                                            completeness = (edited_df.notna().sum().sum() / (len(edited_df) * len(edited_df.columns))) * 100
                                            st.metric("Completeness", f"{completeness:.0f}%")
                                else:
                                    # Enhanced text editor with AI suggestions
                                    render_content_with_images(subcontent, st.session_state.extracted_images)
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        edited_text = st.text_area(
                                            f"Edit {subsection_name}",
                                            value=subcontent,
                                            height=250,
                                            key=f"enhanced_text_{section_name}_{subsection_name}",
                                            help="Use rich text editing with AI assistance"
                                        )
                                        st.session_state.brd_content[section_name][subsection_name] = edited_text
                                    
                                    with col2:
                                        st.write("**AI Assist**")
                                        if st.button(f"✨ Enhance", key=f"enhance_{section_name}_{subsection_name}"):
                                            st.info("AI enhancement coming soon!")
                                        if st.button(f"📝 Summarize", key=f"summary_{section_name}_{subsection_name}"):
                                            st.info("AI summarization coming soon!")
                                
                                st.markdown("---")
                        else:
                            # Single content editing
                            if isinstance(content, pd.DataFrame):
                                st.write("📊 Interactive Table Editor:")
                                edited_df = st.data_editor(
                                    content,
                                    use_container_width=True,
                                    num_rows="dynamic",
                                    column_config={
                                        col: st.column_config.TextColumn(
                                            help=f"Edit {col} values"
                                        ) for col in content.columns
                                    },
                                    key=f"enhanced_table_{section_name}"
                                )
                                st.session_state.brd_content[section_name] = edited_df
                                
                                # Enhanced table insights
                                if not edited_df.empty:
                                    with st.expander("📈 Table Insights"):
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Total Rows", len(edited_df))
                                        with col2:
                                            st.metric("Total Columns", len(edited_df.columns))
                                        with col3:
                                            empty_cells = edited_df.isna().sum().sum()
                                            st.metric("Empty Cells", empty_cells)
                                        with col4:
                                            completeness = ((len(edited_df) * len(edited_df.columns) - empty_cells) / (len(edited_df) * len(edited_df.columns))) * 100
                                            st.metric("Data Completeness", f"{completeness:.1f}%")
                            else:
                                # Enhanced text content editing
                                render_content_with_images(content, st.session_state.extracted_images)
                                
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    edited_text = st.text_area(
                                        f"Edit {section_name}",
                                        value=content,
                                        height=300,
                                        key=f"enhanced_text_{section_name}",
                                        help="Professional text editor with AI assistance"
                                    )
                                    st.session_state.brd_content[section_name] = edited_text
                                
                                with col2:
                                    st.write("**AI Tools**")
                                    if st.button(f"✨ AI Enhance", key=f"ai_enhance_{section_name}"):
                                        st.info("🤖 AI enhancement will improve clarity, completeness, and professional tone")
                                    if st.button(f"🔍 Grammar Check", key=f"grammar_{section_name}"):
                                        st.info("📝 Grammar and style checking coming soon!")
                                    if st.button(f"📊 Add Metrics", key=f"metrics_{section_name}"):
                                        st.info("📈 Smart metric suggestions coming soon!")
        else:
            st.info("👆 Please upload and analyze a document first in the Document Analysis tab")
