import streamlit as st
import os
import json
from google import genai
from google.genai import types
import time
from PIL import Image
import io

st.set_page_config(page_title="Paper to Patient Note", page_icon="📝", layout="wide")

st.markdown("# 📝 Paper to Patient Note")
st.sidebar.header("Handwritten Note Conversion")

st.write(
    """
    This demo uses Gemini 2.5 Flash to convert handwritten medical notes into structured digital text. 
    Upload an image of a handwritten note, and the AI will extract and format the text into proper 
    medical documentation with options for different output formats.
    """
)

# Check for API key
if "GEMINI_API_KEY" not in os.environ:
    st.error("🔑 Please set your GEMINI_API_KEY environment variable to use this demo.")
    st.stop()

@st.cache_resource
def get_gemini_client():
    """Initialize the Gemini client."""
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        return client
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

def create_handwriting_extraction_prompt(output_format, preserve_structure):
    """Create the prompt for handwritten note extraction."""
    
    base_prompt = """You are an expert medical transcriptionist specializing in converting handwritten medical notes to digital text. 

Analyze the provided handwritten medical note image and extract all visible text accurately. Pay special attention to:

1. **Medical terminology**: Correctly identify medical terms, drug names, and abbreviations
2. **Legibility assessment**: Note any unclear or ambiguous text
3. **Structure preservation**: Maintain the original organization and formatting
4. **Completeness**: Extract all visible text including margins, annotations, and corrections

IMPORTANT GUIDELINES:
- Preserve original spelling and grammar (note corrections separately if needed)
- Use standard medical abbreviations when clearly written
- Indicate unclear text with [unclear] or [illegible] markers
- Maintain original paragraph breaks and list structures
- Note any crossed-out or corrected text separately"""

    if output_format == "structured":
        format_instructions = """

STRUCTURED OUTPUT FORMAT:
Return the response in the following JSON format:

{
  "extraction_summary": {
    "legibility_score": "High/Medium/Low",
    "total_words_extracted": 0,
    "unclear_segments": 0,
    "medical_terms_identified": 0,
    "confidence_level": "High/Medium/Low"
  },
  "extracted_text": {
    "main_content": "The primary extracted text content",
    "sections_identified": [
      {
        "section_type": "Chief Complaint/History/Physical Exam/Assessment/Plan/etc.",
        "content": "Text content for this section"
      }
    ],
    "annotations_notes": "Any margin notes, annotations, or additional observations",
    "corrections_crossouts": "Any visible corrections or crossed-out text"
  },
  "text_quality_assessment": {
    "handwriting_quality": "Excellent/Good/Fair/Poor",
    "ink_clarity": "Clear/Faded/Smudged",
    "paper_condition": "Good/Worn/Damaged",
    "overall_readability": "High/Medium/Low"
  },
  "formatting_suggestions": [
    "Suggestions for improving the digital format",
    "Recommendations for structure enhancement"
  ],
  "medical_context": {
    "document_type": "Progress Note/SOAP Note/Consultation/Prescription/etc.",
    "specialty_area": "Identified medical specialty if apparent",
    "key_medical_findings": ["List of important medical information extracted"]
  },
  "transcription_notes": "Additional notes about the transcription process, challenges, or recommendations"
}"""
    else:
        format_instructions = """

RAW TEXT OUTPUT FORMAT:
Return the response as clean, formatted text that preserves the original structure:

EXTRACTED TEXT:
[The complete extracted text content, maintaining original formatting and structure]

TRANSCRIPTION NOTES:
- Legibility Assessment: [High/Medium/Low]
- Confidence Level: [High/Medium/Low] 
- Unclear Segments: [Number and description]
- Special Notes: [Any additional observations about the handwriting, corrections, or content]

QUALITY ASSESSMENT:
- Handwriting Quality: [Assessment]
- Overall Readability: [Assessment]
- Medical Terminology: [Assessment of medical terms identified]"""

    structure_instructions = ""
    if preserve_structure:
        structure_instructions = "\n\nSTRUCTURE PRESERVATION: Maintain the exact formatting, indentation, and organization of the original handwritten note including bullet points, numbering, and spacing."
    else:
        structure_instructions = "\n\nSTRUCTURE IMPROVEMENT: Organize the extracted text into a clean, professional medical note format with proper sections and formatting."

    return base_prompt + format_instructions + structure_instructions

def process_handwritten_note(client, image_data, output_format, preserve_structure):
    """Process handwritten note using Gemini's image understanding."""
    try:
        model = "gemini-2.5-flash-preview-05-20"
        
        # Create the prompt
        prompt = create_handwriting_extraction_prompt(output_format, preserve_structure)
        
        # Prepare content with image
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
                ]
            )
        ]
        
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json" if output_format == "structured" else "text/plain",
        )
        
        # Stream the response
        response_text = ""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        chunks_received = 0
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_text += chunk.text
                chunks_received += 1
                
                # Update progress
                progress = min(chunks_received * 0.15, 0.9)
                progress_bar.progress(progress)
                status_text.text(f"Converting handwritten note to text... {len(response_text)} characters processed")
        
        progress_bar.progress(1.0)
        status_text.text("Conversion complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Parse response based on format
        if output_format == "structured":
            try:
                json_response = json.loads(response_text)
                return json_response
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON response: {str(e)}")
                st.text("Raw response:")
                st.text(response_text[:1000] + "..." if len(response_text) > 1000 else response_text)
                return None
        else:
            return {"raw_text": response_text}
                
    except Exception as e:
        st.error(f"Error processing handwritten note: {str(e)}")
        return None

def display_extraction_results(results, output_format):
    """Display the handwritten note extraction results."""
    if not results:
        return
    
    if output_format == "structured":
        # Display extraction summary
        summary = results.get("extraction_summary", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            legibility = summary.get("legibility_score", "Unknown")
            st.metric("Legibility Score", legibility)
        with col2:
            words = summary.get("total_words_extracted", 0)
            st.metric("Words Extracted", words)
        with col3:
            unclear = summary.get("unclear_segments", 0)
            st.metric("Unclear Segments", unclear)
        with col4:
            confidence = summary.get("confidence_level", "Unknown")
            st.metric("Confidence Level", confidence)
        
        # Main extracted content
        extracted = results.get("extracted_text", {})
        if extracted:
            st.subheader("📄 Extracted Text")
            
            main_content = extracted.get("main_content", "")
            if main_content:
                st.text_area("Main Content", main_content, height=200, disabled=True)
            
            # Sections identified
            sections = extracted.get("sections_identified", [])
            if sections:
                st.subheader("📋 Identified Sections")
                for section in sections:
                    section_type = section.get("section_type", "Unknown Section")
                    content = section.get("content", "")
                    with st.expander(f"{section_type}"):
                        st.write(content)
            
            # Additional content
            annotations = extracted.get("annotations_notes", "")
            if annotations:
                st.subheader("📝 Annotations & Notes")
                st.info(annotations)
            
            corrections = extracted.get("corrections_crossouts", "")
            if corrections:
                st.subheader("✏️ Corrections & Cross-outs")
                st.warning(corrections)
        
        # Quality assessment
        quality = results.get("text_quality_assessment", {})
        if quality:
            st.subheader("🔍 Quality Assessment")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Handwriting Quality:** {quality.get('handwriting_quality', 'Unknown')}")
                st.write(f"**Ink Clarity:** {quality.get('ink_clarity', 'Unknown')}")
            with col2:
                st.write(f"**Paper Condition:** {quality.get('paper_condition', 'Unknown')}")
                st.write(f"**Overall Readability:** {quality.get('overall_readability', 'Unknown')}")
        
        # Medical context
        medical_context = results.get("medical_context", {})
        if medical_context:
            st.subheader("🏥 Medical Context")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Document Type:** {medical_context.get('document_type', 'Unknown')}")
                st.write(f"**Specialty Area:** {medical_context.get('specialty_area', 'General')}")
            
            with col2:
                findings = medical_context.get('key_medical_findings', [])
                if findings:
                    st.write("**Key Medical Findings:**")
                    for finding in findings:
                        st.write(f"• {finding}")
        
        # Formatting suggestions
        suggestions = results.get("formatting_suggestions", [])
        if suggestions:
            st.subheader("💡 Formatting Suggestions")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
        
        # Transcription notes
        if results.get("transcription_notes"):
            st.subheader("📝 Transcription Notes")
            st.info(results["transcription_notes"])
    
    else:
        # Raw text format
        raw_text = results.get("raw_text", "")
        if raw_text:
            st.subheader("📄 Extracted Text")
            st.text_area("Converted Text", raw_text, height=400, disabled=True)
    
    # Download options
    st.subheader("📄 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON download (for structured format)
        if output_format == "structured":
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="📄 Download Full Analysis (JSON)",
                data=json_str,
                file_name=f"handwritten_note_analysis_{int(time.time())}.json",
                mime="application/json"
            )
        else:
            # Text download (for raw format)
            raw_text = results.get("raw_text", "")
            st.download_button(
                label="📄 Download Extracted Text",
                data=raw_text,
                file_name=f"extracted_text_{int(time.time())}.txt",
                mime="text/plain"
            )
    
    with col2:
        # Clean text download
        if output_format == "structured":
            extracted = results.get("extracted_text", {})
            clean_text = extracted.get("main_content", "")
        else:
            # Extract just the main content from raw text
            raw_text = results.get("raw_text", "")
            # Try to extract just the main extracted text portion
            if "EXTRACTED TEXT:" in raw_text:
                parts = raw_text.split("TRANSCRIPTION NOTES:")
                clean_text = parts[0].replace("EXTRACTED TEXT:", "").strip()
            else:
                clean_text = raw_text
        
        if clean_text:
            st.download_button(
                label="📝 Download Clean Text Only",
                data=clean_text,
                file_name=f"clean_extracted_text_{int(time.time())}.txt",
                mime="text/plain"
            )
    
    with col3:
        # Summary report
        if output_format == "structured":
            summary = results.get("extraction_summary", {})
            quality = results.get("text_quality_assessment", {})
            medical_context = results.get("medical_context", {})
            
            summary_report = f"""Handwritten Note Conversion Summary
==========================================

Conversion Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

EXTRACTION SUMMARY:
- Legibility Score: {summary.get('legibility_score', 'Unknown')}
- Total Words Extracted: {summary.get('total_words_extracted', 0)}
- Unclear Segments: {summary.get('unclear_segments', 0)}
- Confidence Level: {summary.get('confidence_level', 'Unknown')}

QUALITY ASSESSMENT:
- Handwriting Quality: {quality.get('handwriting_quality', 'Unknown')}
- Ink Clarity: {quality.get('ink_clarity', 'Unknown')}
- Paper Condition: {quality.get('paper_condition', 'Unknown')}
- Overall Readability: {quality.get('overall_readability', 'Unknown')}

MEDICAL CONTEXT:
- Document Type: {medical_context.get('document_type', 'Unknown')}
- Specialty Area: {medical_context.get('specialty_area', 'General')}

TRANSCRIPTION NOTES:
{results.get('transcription_notes', 'None')}
"""
        else:
            summary_report = f"""Handwritten Note Conversion Summary
==========================================

Conversion Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Output Format: Raw Text

Extracted Content:
{results.get('raw_text', 'No content extracted')}
"""
        
        st.download_button(
            label="📋 Download Summary Report",
            data=summary_report,
            file_name=f"handwriting_conversion_summary_{int(time.time())}.txt",
            mime="text/plain"
        )

# Initialize client
client = get_gemini_client()
if not client:
    st.stop()

# Main interface
st.subheader("Upload Handwritten Medical Note")
st.write("Upload an image of a handwritten medical note. Supported formats: PNG, JPG, JPEG, WEBP, HEIC, HEIF")

# File upload
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg', 'webp', 'heic', 'heif'],
    help="Upload a clear image of the handwritten note for best results. Maximum file size: 20MB"
)

# Configuration options
st.subheader("Conversion Options")

col1, col2 = st.columns(2)

with col1:
    output_format = st.selectbox(
        "Output Format",
        options=["structured", "raw_text"],
        format_func=lambda x: "Structured JSON Analysis" if x == "structured" else "Raw Text Output",
        help="Choose between detailed structured analysis or simple text extraction"
    )

with col2:
    preserve_structure = st.checkbox(
        "Preserve Original Structure",
        value=True,
        help="Maintain the original formatting and layout vs. reformatting for clarity"
    )

# Display uploaded image
if uploaded_file is not None:
    # Display image preview
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📷 Uploaded Image Preview")
        st.image(image, caption="Handwritten Note", use_column_width=True)
    
    with col2:
        st.subheader("📊 Image Information")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
        st.write(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"**Format:** {image.format}")
    
    # Process button
    if st.button("🔤 Convert Handwritten Note to Text", type="primary"):
        with st.spinner("Converting handwritten note to digital text... This may take a moment."):
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            # Convert to RGB if necessary (for JPEG compatibility)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            
            result = process_handwritten_note(client, img_byte_arr, output_format, preserve_structure)
            
            if result:
                st.session_state.handwriting_result = result
                st.session_state.handwriting_format = output_format
                st.success("✅ Handwritten note conversion completed!")

# Display results
if hasattr(st.session_state, 'handwriting_result') and st.session_state.handwriting_result:
    st.markdown("---")
    st.subheader("🎯 Conversion Results")
    
    display_extraction_results(st.session_state.handwriting_result, st.session_state.handwriting_format)
    
    # Clear results button
    if st.button("🗑️ Clear Results"):
        if 'handwriting_result' in st.session_state:
            del st.session_state.handwriting_result
        if 'handwriting_format' in st.session_state:
            del st.session_state.handwriting_format
        st.rerun()

# Sidebar information
with st.sidebar:
    st.markdown("### 📝 How It Works")
    st.markdown("""
    1. **📷 Upload Image**: Take or upload photo of handwritten note
    2. **⚙️ Configure Options**: Choose output format and structure
    3. **🧠 AI Processing**: Gemini analyzes and extracts text
    4. **📄 Review Results**: Check extracted text and quality metrics
    5. **💾 Download**: Export in multiple formats
    """)
    
    st.markdown("### 🔧 Features")
    st.markdown("""
    - **🤖 Advanced OCR**: AI-powered handwriting recognition
    - **🏥 Medical Focus**: Specialized for medical terminology
    - **📊 Quality Assessment**: Legibility and confidence scoring
    - **📋 Structure Detection**: Identifies note sections automatically
    - **✏️ Correction Tracking**: Notes crossed-out or corrected text
    - **📄 Multiple Formats**: JSON analysis or clean text output
    """)
    
    st.markdown("### 📷 Image Guidelines")
    st.markdown("""
    **For Best Results:**
    - Use good lighting conditions
    - Ensure text is clearly visible
    - Avoid shadows and glare
    - Keep image straight and stable
    - Include full page/note boundaries
    - Use high resolution when possible
    """)
    
    st.markdown("### 📋 Supported Content")
    st.markdown("""
    **Works Well With:**
    - Progress notes and SOAP notes
    - Prescription pads
    - Consultation notes
    - Physical exam findings
    - Medical history documentation
    - Nursing notes
    - Discharge summaries
    """)
    
    st.markdown("### 🏥 Use Cases")
    st.markdown("""
    - **Digital Conversion**: Modernize paper-based records
    - **Backup Documentation**: Create digital copies
    - **Text Search**: Make handwritten notes searchable
    - **EHR Integration**: Prepare text for electronic systems
    - **Quality Improvement**: Standardize documentation
    - **Accessibility**: Improve readability of notes
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Write clearly and legibly
    - Use standard medical abbreviations
    - Avoid excessive corrections
    - Ensure adequate spacing
    - Use dark ink on light paper
    - Keep consistent writing size
    """)
    
    st.markdown("### ⚠️ Important Notice")
    st.warning("""
    **AI-Assisted Transcription Tool**
    
    Always verify extracted text for:
    - Medical accuracy and completeness
    - Correct drug names and dosages
    - Proper patient information
    - Critical clinical details
    
    Review and validate all AI-generated content before clinical use.
    """)
    
    st.markdown("### 🔗 Resources")
    st.markdown("""
    - [Medical Documentation Standards](https://www.jointcommission.org/standards/)
    - [EHR Best Practices](https://www.healthit.gov/topic/health-it-basics/electronic-health-records-ehrs)
    - [HIPAA Compliance Guidelines](https://www.hhs.gov/hipaa/index.html)
    """) 