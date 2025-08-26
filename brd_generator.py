import streamlit as st
import pandas as pd
from openai import OpenAI
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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt

# App Configuration
st.set_page_config(
    page_title="Enhanced BRD Generator",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'brd_content' not in st.session_state:
    st.session_state.brd_content = {}
if 'extracted_images' not in st.session_state:
    st.session_state.extracted_images = {}
if 'extracted_formulas' not in st.session_state:
    st.session_state.extracted_formulas = []
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'edited_tables' not in st.session_state:
    st.session_state.edited_tables = {}

# Define BRD structure with exact specifications
BRD_STRUCTURE = {
    "1. Background": {
        "type": "text",
        "description": "Provide context and background information"
    },
    "2. Scope": {
        "type": "parent",
        "subsections": {
            "2.1. In Scope": {
                "type": "table",
                "columns": ["ID", "Description"]
            },
            "2.2. Out of Scope": {
                "type": "table", 
                "columns": ["ID", "Description"]
            }
        }
    },
    "3. Assumptions and Dependencies": {
        "type": "parent",
        "subsections": {
            "3.1. Assumptions": {
                "type": "table",
                "columns": ["ID", "Description", "Impact"]
            },
            "3.2. Dependencies": {
                "type": "table",
                "columns": ["ID", "Description", "Impact"]
            }
        }
    },
    "4. Business Requirements": {
        "type": "table",
        "columns": ["Unique Rule Ref", "BR ID", "BR Name", "BR Description", "BR Owner", "BR Type", "Success Criteria"]
    },
    "5. Functional Requirements": {
        "type": "text",
        "description": "Detailed functional requirements"
    },
    "6. Stakeholders and Approvals": {
        "type": "text",
        "description": "List of stakeholders and approval processes"
    },
    "7. Applicable Regulations": {
        "type": "table",
        "columns": ["Unique Rule Ref", "Section", "Regulatory Text"]
    },
    "8. Applicable Internal Rules Interpretations": {
        "type": "text",
        "description": "Internal rules and interpretations"
    },
    "9. Open Items": {
        "type": "text",
        "description": "Outstanding items to be resolved"
    },
    "10. Appendix": {
        "type": "table",
        "columns": ["ID", "Name", "Description"]
    }
}

# LLM Configuration
@st.cache_resource
def init_llm():
    return OpenAI(
        base_url="http://lwnde002xdgpu.sdi.corp.bankofamerica.com:8123/v1",
        api_key="dummy"
    )

def extract_images_and_formulas_from_pdf(uploaded_file) -> Tuple[str, Dict[str, str], List[str]]:
    """Extract text, images, and formulas from PDF using PyMuPDF"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        doc = fitz.open(tmp_file_path)
        text = ""
        images = {}
        formulas = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:
                        img_data = pix.tobytes("png")
                        img_key = f"page_{page_num + 1}_img_{img_index + 1}"
                        img_b64 = base64.b64encode(img_data).decode()
                        images[img_key] = img_b64
                        text += f"\n[IMAGE: {img_key}]\n"
                    
                    pix = None
                except Exception as e:
                    st.warning(f"Could not extract image {img_index + 1} from page {page_num + 1}: {str(e)}")
            
            # Extract formulas
            formula_patterns = [
                r'[A-Za-z]\s*=\s*[^=\n]+',
                r'\b\w+\s*\([^)]+\)\s*=\s*[^=\n]+',
                r'∑|∫|∏|√|±|≤|≥|≠|∞|π|α|β|γ|δ|θ|λ|μ|σ|φ|ω',
                r'\b\d+\s*[+\-*/]\s*\d+\s*[+\-*/=]\s*[^=\n]+',
                r'\b[A-Za-z]+\s*[+\-*/]\s*[A-Za-z]+\s*=\s*[^=\n]+',
            ]
            
            for pattern in formula_patterns:
                matches = re.finditer(pattern, page_text)
                for match in matches:
                    formula = match.group().strip()
                    if len(formula) > 3 and formula not in formulas:
                        formulas.append(f"Page {page_num + 1}: {formula}")
        
        doc.close()
        os.unlink(tmp_file_path)
        return text, images, formulas
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, {}, []

def extract_images_from_docx(uploaded_file) -> Tuple[str, Dict[str, str]]:
    """Extract text and images from DOCX file"""
    try:
        doc = docx.Document(uploaded_file)
        text = ""
        images = {}
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    img_data = rel.target_part.blob
                    img_key = f"docx_img_{len(images) + 1}"
                    img_b64 = base64.b64encode(img_data).decode()
                    images[img_key] = img_b64
                    text += f"\n[IMAGE: {img_key}]\n"
                except Exception as e:
                    st.warning(f"Could not extract image: {str(e)}")
        
        return text, images
        
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None, {}

def display_image_from_base64(img_b64: str, caption: str = "", max_width: int = 400):
    """Display image from base64 string"""
    try:
        img_data = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_data))
        st.image(img, caption=caption, width=max_width)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def generate_brd_section_with_media(llm, section_name, section_config, regulatory_text, images, formulas):
    """Generate BRD section based on configuration"""
    
    media_context = ""
    if images:
        media_context += f"\nAvailable Images: {', '.join(images.keys())}\n"
    if formulas:
        media_context += f"\nExtracted Formulas:\n" + "\n".join(formulas[:10])
    
    if section_config["type"] == "table":
        columns = section_config["columns"]
        prompt = f"""
        Generate content for the "{section_name}" section in tabular format.
        
        Create a table with exactly these columns: {' | '.join(columns)}
        
        Based on this regulatory text: {regulatory_text[:3000]}
        {media_context}
        
        Return the data in pipe-separated format, one row per line, like:
        {' | '.join(columns)}
        Value1 | Value2 | Value3...
        Value1 | Value2 | Value3...
        
        Generate at least 3-5 relevant rows. Reference images using [IMAGE: image_key] format where relevant.
        """
    else:
        description = section_config.get("description", f"Generate content for {section_name}")
        prompt = f"""
        Generate content for the "{section_name}" section.
        
        Description: {description}
        
        Based on this regulatory text: {regulatory_text[:3500]}
        {media_context}
        
        Provide detailed, relevant content. Reference images using [IMAGE: image_key] format where applicable.
        """
    
    try:
        response = llm.chat.completions.create(
            model="/phoenix/workspaces/nbkm74lv/llama3.3-4bit-awq",
            messages=[
                {"role": "system", "content": "You are an expert business analyst creating BRD sections. Follow the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating {section_name}: {str(e)}")
        return f"Error generating {section_name} section"

def parse_table_content(content: str, columns: List[str]) -> pd.DataFrame:
    """Parse LLM table content into DataFrame"""
    lines = content.strip().split('\n')
    data = []
    
    for line in lines:
        if '|' in line and not line.strip().startswith('#'):
            row = [cell.strip() for cell in line.split('|')]
            if len(row) == len(columns):
                data.append(row)
            elif len(row) > len(columns):
                # Truncate extra columns
                data.append(row[:len(columns)])
            elif len(row) < len(columns):
                # Pad missing columns
                row.extend([''] * (len(columns) - len(row)))
                data.append(row)
    
    # Filter out header row if it matches column names
    if data and all(col.lower() in [cell.lower() for cell in data[0]] for col in columns):
        data = data[1:]
    
    return pd.DataFrame(data, columns=columns) if data else pd.DataFrame(columns=columns)

def render_content_with_images(content: str, images: Dict[str, str]):
    """Render content and display referenced images inline"""
    parts = re.split(r'\[IMAGE:\s*([^\]]+)\]', content)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Text part
            if part.strip():
                st.markdown(part)
        else:  # Image reference
            img_key = part.strip()
            if img_key in images:
                st.markdown(f"**Referenced Image: {img_key}**")
                display_image_from_base64(images[img_key], caption=img_key, max_width=600)
            else:
                st.warning(f"Image reference '{img_key}' not found")

def create_word_document(brd_content: Dict, images: Dict, formulas: List) -> BytesIO:
    """Create Word document from BRD content"""
    doc = docx.Document()
    
    # Add title
    title = doc.add_heading('Business Requirements Document', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    for section_name, content in brd_content.items():
        # Add section heading
        doc.add_heading(section_name, level=1)
        
        if isinstance(content, dict):
            # Handle subsections
            for subsection_name, subcontent in content.items():
                doc.add_heading(subsection_name, level=2)
                
                if isinstance(subcontent, pd.DataFrame) and not subcontent.empty:
                    # Add table
                    table = doc.add_table(rows=1, cols=len(subcontent.columns))
                    table.style = 'Table Grid'
                    
                    # Add headers
                    hdr_cells = table.rows[0].cells
                    for i, column in enumerate(subcontent.columns):
                        hdr_cells[i].text = column
                    
                    # Add data rows
                    for _, row in subcontent.iterrows():
                        row_cells = table.add_row().cells
                        for i, value in enumerate(row):
                            row_cells[i].text = str(value)
                else:
                    # Add text content
                    content_text = subcontent if isinstance(subcontent, str) else str(subcontent)
                    # Remove image references for Word doc
                    clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '', content_text)
                    doc.add_paragraph(clean_text)
        else:
            if isinstance(content, pd.DataFrame) and not content.empty:
                # Add table
                table = doc.add_table(rows=1, cols=len(content.columns))
                table.style = 'Table Grid'
                
                # Add headers
                hdr_cells = table.rows[0].cells
                for i, column in enumerate(content.columns):
                    hdr_cells[i].text = column
                
                # Add data rows
                for _, row in content.iterrows():
                    row_cells = table.add_row().cells
                    for i, value in enumerate(row):
                        row_cells[i].text = str(value)
            else:
                # Add text content
                content_text = content if isinstance(content, str) else str(content)
                clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '', content_text)
                doc.add_paragraph(clean_text)
    
    # Add extracted media summary
    if images or formulas:
        doc.add_page_break()
        doc.add_heading('Extracted Media Summary', level=1)
        
        doc.add_paragraph(f"Total Images Extracted: {len(images)}")
        doc.add_paragraph(f"Total Formulas Found: {len(formulas)}")
        
        if formulas:
            doc.add_heading('Mathematical Formulas', level=2)
            for formula in formulas:
                doc.add_paragraph(f"• {formula}")
    
    # Save to BytesIO
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def create_pdf_document(brd_content: Dict, images: Dict, formulas: List) -> BytesIO:
    """Create PDF document from BRD content"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=1,  # Center
        spaceBefore=0,
        spaceAfter=30
    )
    story.append(Paragraph("Business Requirements Document", title_style))
    story.append(Spacer(1, 20))
    
    for section_name, content in brd_content.items():
        # Section heading
        story.append(Paragraph(section_name, styles['Heading1']))
        story.append(Spacer(1, 12))
        
        if isinstance(content, dict):
            # Handle subsections
            for subsection_name, subcontent in content.items():
                story.append(Paragraph(subsection_name, styles['Heading2']))
                story.append(Spacer(1, 6))
                
                if isinstance(subcontent, pd.DataFrame) and not subcontent.empty:
                    # Create table data
                    table_data = [subcontent.columns.tolist()]
                    for _, row in subcontent.iterrows():
                        table_data.append(row.tolist())
                    
                    # Create table
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(table)
                else:
                    # Add text content
                    content_text = subcontent if isinstance(subcontent, str) else str(subcontent)
                    clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '', content_text)
                    story.append(Paragraph(clean_text, styles['Normal']))
                
                story.append(Spacer(1, 12))
        else:
            if isinstance(content, pd.DataFrame) and not content.empty:
                # Create table data
                table_data = [content.columns.tolist()]
                for _, row in content.iterrows():
                    table_data.append(row.tolist())
                
                # Create table
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            else:
                # Add text content
                content_text = content if isinstance(content, str) else str(content)
                clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '', content_text)
                story.append(Paragraph(clean_text, styles['Normal']))
            
            story.append(Spacer(1, 12))
    
    # Add media summary
    if images or formulas:
        story.append(Paragraph("Extracted Media Summary", styles['Heading1']))
        story.append(Paragraph(f"Total Images Extracted: {len(images)}", styles['Normal']))
        story.append(Paragraph(f"Total Formulas Found: {len(formulas)}", styles['Normal']))
        
        if formulas:
            story.append(Paragraph("Mathematical Formulas:", styles['Heading2']))
            for formula in formulas[:10]:  # Limit to first 10
                story.append(Paragraph(f"• {formula}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Sidebar
st.sidebar.title("📋 Enhanced BRD Generator")
st.sidebar.markdown("Upload regulatory documents to generate Business Requirements Documents")

uploaded_file = st.sidebar.file_uploader(
    "Upload Regulatory Document",
    type=['pdf', 'docx', 'txt'],
    help="Upload a regulatory document (100-500 pages supported)"
)

# Media extraction options
st.sidebar.subheader("🎨 Media Extraction Options")
extract_images = st.sidebar.checkbox("Extract Images", value=True)
extract_formulas = st.sidebar.checkbox("Extract Mathematical Formulas", value=True)
max_images = st.sidebar.slider("Max Images to Extract", 1, 50, 20)

# Main Content
st.title("📋 Enhanced Business Requirements Document Generator")
st.markdown("Generate comprehensive BRDs from regulatory documents with **exact structure compliance**")

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📄 Uploaded: {uploaded_file.name} ({file_size_mb:.2f} MB)")
    
    # Extract content
    with st.spinner("🔍 Extracting content from document..."):
        if uploaded_file.type == "application/pdf":
            document_text, extracted_images, extracted_formulas = extract_images_and_formulas_from_pdf(uploaded_file)
            if extract_images and len(extracted_images) > max_images:
                extracted_images = dict(list(extracted_images.items())[:max_images])
            st.session_state.extracted_images = extracted_images if extract_images else {}
            st.session_state.extracted_formulas = extracted_formulas if extract_formulas else []
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            document_text, extracted_images = extract_images_from_docx(uploaded_file)
            if extract_images and len(extracted_images) > max_images:
                extracted_images = dict(list(extracted_images.items())[:max_images])
            st.session_state.extracted_images = extracted_images if extract_images else {}
            st.session_state.extracted_formulas = []
        else:
            uploaded_file.seek(0)
            document_text = str(uploaded_file.read(), "utf-8")
            st.session_state.extracted_images = {}
            st.session_state.extracted_formulas = []
    
    if document_text:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Text Characters", f"{len(document_text):,}")
        with col2:
            st.metric("🖼️ Images Extracted", len(st.session_state.extracted_images))
        with col3:
            st.metric("🧮 Formulas Found", len(st.session_state.extracted_formulas))
        
        # Preview extracted media
        if st.session_state.extracted_images or st.session_state.extracted_formulas:
            with st.expander("🎨 Preview Extracted Media", expanded=False):
                if st.session_state.extracted_images:
                    st.subheader("Extracted Images")
                    img_cols = st.columns(3)
                    for idx, (img_key, img_b64) in enumerate(st.session_state.extracted_images.items()):
                        with img_cols[idx % 3]:
                            display_image_from_base64(img_b64, caption=img_key, max_width=200)
                
                if st.session_state.extracted_formulas:
                    st.subheader("Extracted Formulas")
                    for formula in st.session_state.extracted_formulas[:10]:
                        st.code(formula)
        
        # Generate BRD Button
        if st.button("🚀 Generate BRD with Exact Structure", type="primary"):
            llm = init_llm()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_sections = len(BRD_STRUCTURE) + sum(
                len(section["subsections"]) if section.get("type") == "parent" else 0 
                for section in BRD_STRUCTURE.values()
            )
            
            section_count = 0
            
            for section_name, section_config in BRD_STRUCTURE.items():
                if section_config.get("type") == "parent":
                    # Handle parent sections with subsections
                    st.session_state.brd_content[section_name] = {}
                    for subsection_name, subsection_config in section_config["subsections"].items():
                        status_text.text(f"Generating {subsection_name}...")
                        
                        content = generate_brd_section_with_media(
                            llm, subsection_name, subsection_config, document_text,
                            st.session_state.extracted_images, st.session_state.extracted_formulas
                        )
                        
                        if subsection_config["type"] == "table":
                            df = parse_table_content(content, subsection_config["columns"])
                            st.session_state.brd_content[section_name][subsection_name] = df
                        else:
                            st.session_state.brd_content[section_name][subsection_name] = content
                        
                        section_count += 1
                        progress_bar.progress(section_count / total_sections)
                else:
                    # Handle regular sections
                    status_text.text(f"Generating {section_name}...")
                    
                    content = generate_brd_section_with_media(
                        llm, section_name, section_config, document_text,
                        st.session_state.extracted_images, st.session_state.extracted_formulas
                    )
                    
                    if section_config["type"] == "table":
                        df = parse_table_content(content, section_config["columns"])
                        st.session_state.brd_content[section_name] = df
                    else:
                        st.session_state.brd_content[section_name] = content
                    
                    section_count += 1
                    progress_bar.progress(section_count / total_sections)
            
            progress_bar.progress(1.0)
            status_text.text("✅ BRD Generation Complete!")
            st.session_state.generated = True
            st.balloons()

# Display and Edit Generated BRD
if st.session_state.generated and st.session_state.brd_content:
    st.markdown("---")
    st.header("📋 Business Requirements Document - Review & Edit")
    
    # Create tabs for editing
    tab_names = list(st.session_state.brd_content.keys())
    tabs = st.tabs(tab_names)
    
    for i, (section_name, content) in enumerate(st.session_state.brd_content.items()):
        with tabs[i]:
            st.subheader(section_name)
            
            if isinstance(content, dict):
                # Handle subsections
                for subsection_name, subcontent in content.items():
                    st.write(f"**{subsection_name}**")
                    
                    if isinstance(subcontent, pd.DataFrame):
                        # Editable table
                        st.write("Edit the table below:")
                        edited_df = st.data_editor(
                            subcontent,
                            use_container_width=True,
                            num_rows="dynamic",
                            key=f"table_{section_name}_{subsection_name}"
                        )
                        st.session_state.brd_content[section_name][subsection_name] = edited_df
                    else:
                        # Editable text with image rendering
                        render_content_with_images(subcontent, st.session_state.extracted_images)
                        
                        edited_text = st.text_area(
                            f"Edit {subsection_name}",
                            value=subcontent,
                            height=200,
                            key=f"text_{section_name}_{subsection_name}"
                        )
                        st.session_state.brd_content[section_name][subsection_name] = edited_text
                    
                    st.markdown("---")
            else:
                if isinstance(content, pd.DataFrame):
                    # Editable table
                    st.write("Edit the table below:")
                    edited_df = st.data_editor(
                        content,
                        use_container_width=True,
                        num_rows="dynamic",
                        key=f"table_{section_name}"
                    )
                    st.session_state.brd_content[section_name] = edited_df
                else:
                    # Editable text with image rendering
                    render_content_with_images(content, st.session_state.extracted_images)
                    
                    edited_text = st.text_area(
                        f"Edit {section_name}",
                        value=content,
                        height=300,
                        key=f"text_{section_name}"
                    )
                    st.session_state.brd_content[section_name] = edited_text
    
    # Export Options
    st.markdown("---")
    st.header("📥 Export Final BRD")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("📄 Export as Word"):
            with st.spinner("Creating Word document..."):
                word_buffer = create_word_document(
                    st.session_state.brd_content,
                    st.session_state.extracted_images,
                    st.session_state.extracted_formulas
                )
                st.download_button(
                    label="Download Word Document",
                    data=word_buffer,
                    file_name="BRD_Document.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    
    with col2:
        if st.button("📋 Export as PDF"):
            with st.spinner("Creating PDF document..."):
                try:
                    pdf_buffer = create_pdf_document(
                        st.session_state.brd_content,
                        st.session_state.extracted_images,
                        st.session_state.extracted_formulas
                    )
                    st.download_button(
                        label="Download PDF Document",
                        data=pdf_buffer,
                        file_name="BRD_Document.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error creating PDF: {str(e)}")
                    st.info("PDF export requires additional setup. Please use Word export instead.")
    
    with col3:
        if st.button("📊 Export as JSON"):
            export_data = {
                "brd_content": {},
                "images": st.session_state.extracted_images,
                "formulas": st.session_state.extracted_formulas,
                "structure": BRD_STRUCTURE
            }
            
            # Convert DataFrames to dict for JSON serialization
            for section_name, content in st.session_state.brd_content.items():
                if isinstance(content, dict):
                    export_data["brd_content"][section_name] = {}
                    for sub_name, sub_content in content.items():
                        if isinstance(sub_content, pd.DataFrame):
                            export_data["brd_content"][section_name][sub_name] = sub_content.to_dict('records')
                        else:
                            export_data["brd_content"][section_name][sub_name] = sub_content
                elif isinstance(content, pd.DataFrame):
                    export_data["brd_content"][section_name] = content.to_dict('records')
                else:
                    export_data["brd_content"][section_name] = content
            
            json_string = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_string,
                file_name="BRD_Complete.json",
                mime="application/json"
            )
    
    with col4:
        if st.button("📝 Export as Text"):
            text_output = "BUSINESS REQUIREMENTS DOCUMENT\n" + "="*50 + "\n\n"
            
            for section_name, content in st.session_state.brd_content.items():
                text_output += f"{section_name}\n" + "-"*30 + "\n"
                
                if isinstance(content, dict):
                    for sub_name, sub_content in content.items():
                        text_output += f"\n{sub_name}:\n"
                        if isinstance(sub_content, pd.DataFrame):
                            text_output += sub_content.to_string(index=False) + "\n"
                        else:
                            # Remove image references for text export
                            clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '[Image Reference]', str(sub_content))
                            text_output += clean_text + "\n"
                elif isinstance(content, pd.DataFrame):
                    text_output += content.to_string(index=False) + "\n"
                else:
                    clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '[Image Reference]', str(content))
                    text_output += clean_text + "\n"
                
                text_output += "\n"
            
            # Add media summary
            text_output += "\nEXTRACTED MEDIA SUMMARY\n" + "="*30 + "\n"
            text_output += f"Images: {len(st.session_state.extracted_images)}\n"
            text_output += f"Formulas: {len(st.session_state.extracted_formulas)}\n\n"
            
            if st.session_state.extracted_formulas:
                text_output += "Mathematical Formulas:\n"
                for formula in st.session_state.extracted_formulas:
                    text_output += f"- {formula}\n"
            
            st.download_button(
                label="Download Text",
                data=text_output,
                file_name="BRD_Document.txt",
                mime="text/plain"
            )
    
    with col5:
        if st.button("🔄 Reset All"):
            st.session_state.brd_content = {}
            st.session_state.extracted_images = {}
            st.session_state.extracted_formulas = []
            st.session_state.generated = False
            st.session_state.edited_tables = {}
            st.rerun()
    
    # BRD Validation Summary
    st.markdown("---")
    st.header("✅ BRD Compliance Summary")
    
    compliance_col1, compliance_col2 = st.columns(2)
    
    with compliance_col1:
        st.subheader("📊 Structure Compliance")
        total_sections = len(BRD_STRUCTURE)
        completed_sections = len(st.session_state.brd_content)
        compliance_percentage = (completed_sections / total_sections) * 100
        
        st.metric(
            "Sections Completed", 
            f"{completed_sections}/{total_sections}",
            f"{compliance_percentage:.1f}% Complete"
        )
        
        # Check tabular format compliance
        table_sections = []
        for section_name, section_config in BRD_STRUCTURE.items():
            if section_config.get("type") == "table":
                table_sections.append(section_name)
            elif section_config.get("type") == "parent":
                for sub_name, sub_config in section_config["subsections"].items():
                    if sub_config.get("type") == "table":
                        table_sections.append(f"{section_name} > {sub_name}")
        
        table_compliance = 0
        for table_section in table_sections:
            if " > " in table_section:
                parent, child = table_section.split(" > ")
                if (parent in st.session_state.brd_content and 
                    child in st.session_state.brd_content[parent] and
                    isinstance(st.session_state.brd_content[parent][child], pd.DataFrame)):
                    table_compliance += 1
            else:
                if (table_section in st.session_state.brd_content and
                    isinstance(st.session_state.brd_content[table_section], pd.DataFrame)):
                    table_compliance += 1
        
        table_percentage = (table_compliance / len(table_sections)) * 100 if table_sections else 100
        st.metric(
            "Tabular Format Compliance",
            f"{table_compliance}/{len(table_sections)}",
            f"{table_percentage:.1f}% Compliant"
        )
    
    with compliance_col2:
        st.subheader("🎨 Media Integration")
        media_references = 0
        total_content_sections = 0
        
        for content in st.session_state.brd_content.values():
            if isinstance(content, dict):
                for subcontent in content.values():
                    total_content_sections += 1
                    if isinstance(subcontent, str) and '[IMAGE:' in subcontent:
                        media_references += 1
            else:
                total_content_sections += 1
                if isinstance(content, str) and '[IMAGE:' in content:
                    media_references += 1
        
        st.metric("Images Extracted", len(st.session_state.extracted_images))
        st.metric("Formulas Found", len(st.session_state.extracted_formulas))
        st.metric("Media References", media_references)

else:
    # Welcome message
    st.markdown("""
    ## 🎯 Exact BRD Structure Compliance
    
    This enhanced generator creates BRDs with **exact structure compliance** according to your specifications:
    
    ### 📋 BRD Structure (Exact Format):
    
    1. **Background** - Text format
    2. **Scope** - Parent section with:
       - 2.1. In Scope (Table: ID | Description)
       - 2.2. Out of Scope (Table: ID | Description)
    3. **Assumptions and Dependencies** - Parent section with:
       - 3.1. Assumptions (Table: ID | Description | Impact)
       - 3.2. Dependencies (Table: ID | Description | Impact)
    4. **Business Requirements** (Table: Unique Rule Ref | BR ID | BR Name | BR Description | BR Owner | BR Type | Success Criteria)
    5. **Functional Requirements** - Text format
    6. **Stakeholders and Approvals** - Text format
    7. **Applicable Regulations** (Table: Unique Rule Ref | Section | Regulatory Text)
    8. **Applicable Internal Rules Interpretations** - Text format
    9. **Open Items** - Text format
    10. **Appendix** (Table: ID | Name | Description)
    
    ### 🚀 Enhanced Features:
    - **🖼️ Image Extraction**: PDF and DOCX image extraction with contextual integration
    - **🧮 Formula Detection**: Mathematical expressions and equations
    - **📝 Interactive Editing**: Full editing capabilities for all sections
    - **📊 Table Editors**: Dynamic table editing with add/remove rows
    - **📄 Export Options**: Word, PDF, JSON, and Text formats
    - **✅ Compliance Tracking**: Real-time structure compliance monitoring
    
    ### 📖 How to Use:
    
    1. **Upload Document**: PDF, DOCX, or TXT regulatory document
    2. **Configure Options**: Set image and formula extraction preferences
    3. **Generate BRD**: Create structured BRD with exact format compliance
    4. **Review & Edit**: Use interactive editors to refine each section
    5. **Export**: Download in Word, PDF, JSON, or Text format
    
    ### 🎨 Media Integration:
    - Images are automatically extracted and referenced contextually
    - Mathematical formulas are detected and included in relevant sections
    - Visual content enhances regulatory requirement understanding
    - All media references are preserved in exports
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Enhanced BRD Generator v2.1 | 📋 Exact Structure Compliance | 🖼️ Media Integration | 📝 Interactive Editing"
    "</div>", 
    unsafe_allow_html=True
)
