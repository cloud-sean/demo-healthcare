import streamlit as st
import base64
import os
import json
from google import genai
from google.genai import types
import time
import pandas as pd

st.set_page_config(page_title="ICD Code Extraction", page_icon="🔍", layout="wide")

st.markdown("# 🔍 ICD Code Extraction")
st.sidebar.header("ICD Code Extraction Demo")

st.write(
    """
    This demo uses Gemini 2.5 Flash to analyze medical audio or text and automatically extract 
    relevant ICD-10/11 codes with supporting evidence. Perfect for medical coding assistance 
    and clinical documentation review.
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

def icd_extraction_prompt():
    """Return the ICD code extraction prompt with JSON schema."""
    return """You are a medical coding specialist. Analyze the provided medical content and extract relevant ICD-10/11 codes with supporting evidence.

Please identify:
1. **Primary Diagnoses**: Main conditions mentioned
2. **Secondary Diagnoses**: Additional conditions or comorbidities
3. **Symptoms**: Signs and symptoms that may require coding
4. **Procedures**: Any medical procedures mentioned (if applicable)

For each identified code, provide:
- The specific ICD-10 or ICD-11 code
- Description of the condition/procedure
- Direct quote from the text as evidence
- Confidence level (High/Medium/Low)
- Category (Primary Diagnosis, Secondary Diagnosis, Symptom, Procedure)

Return the response in the following JSON format:

{
  "summary": {
    "total_codes": 0,
    "primary_diagnoses": 0,
    "secondary_diagnoses": 0,
    "symptoms": 0,
    "procedures": 0
  },
  "codes": [
    {
      "code": "ICD-10 or ICD-11 code",
      "description": "Full description of the condition/procedure",
      "evidence": "Direct quote from the audio/text that supports this code",
      "confidence": "High/Medium/Low",
      "category": "Primary Diagnosis/Secondary Diagnosis/Symptom/Procedure",
      "additional_notes": "Any relevant clinical notes or considerations"
    }
  ],
  "clinical_summary": "Brief summary of the medical content analyzed",
  "coding_notes": "Any important notes about the coding decisions or areas needing clarification"
}

Be precise and conservative in your coding recommendations. Only suggest codes that have clear supporting evidence in the provided content."""

def get_audio_mime_type(filename):
    """Get MIME type for audio file."""
    extension = filename.lower().split('.')[-1]
    mime_types = {
        'wav': 'audio/wav',
        'mp3': 'audio/mpeg',
        'm4a': 'audio/m4a',
        'flac': 'audio/flac',
        'ogg': 'audio/ogg',
        'webm': 'audio/webm'
    }
    return mime_types.get(extension, 'audio/wav')

def extract_icd_codes(client, audio_file=None, text_content=None, content_type="audio"):
    """Extract ICD codes using Gemini with JSON output."""
    try:
        model = "gemini-2.5-flash-preview-05-20"
        
        # Prepare content based on input type
        parts = []
        parts.append(types.Part.from_text(text=icd_extraction_prompt()))
        
        if content_type == "audio" and audio_file:
            # Read audio file and create proper audio part
            audio_bytes = audio_file.read()
            mime_type = get_audio_mime_type(audio_file.name)
            
            # Create audio part for Gemini
            parts.append(types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type
            ))
            
        elif content_type == "text" and text_content:
            parts.append(types.Part.from_text(text=f"Medical text to analyze for ICD codes:\n\n{text_content}"))
            
        elif content_type == "both":
            if audio_file:
                audio_bytes = audio_file.read()
                mime_type = get_audio_mime_type(audio_file.name)
                parts.append(types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                ))
            
            if text_content:
                parts.append(types.Part.from_text(text=f"Additional text context:\n\n{text_content}"))
        
        contents = [
            types.Content(
                role="user",
                parts=parts
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",  # Request JSON output
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
                
                # Update progress (estimated)
                progress = min(chunks_received * 0.15, 0.9)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing medical content... {len(response_text)} characters generated")
        
        progress_bar.progress(1.0)
        status_text.text("Analysis complete!")
        time.sleep(0.5)  # Brief pause to show completion
        progress_bar.empty()
        status_text.empty()
        
        # Parse JSON response
        try:
            json_response = json.loads(response_text)
            return json_response
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {str(e)}")
            st.text("Raw response:")
            st.text(response_text)
            return None
                
    except Exception as e:
        st.error(f"Error extracting ICD codes: {str(e)}")
        return None

def display_icd_results(results):
    """Display ICD code extraction results in an organized format."""
    if not results:
        return
    
    # Display summary
    summary = results.get("summary", {})
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Codes", summary.get("total_codes", 0))
    with col2:
        st.metric("Primary Diagnoses", summary.get("primary_diagnoses", 0))
    with col3:
        st.metric("Secondary Diagnoses", summary.get("secondary_diagnoses", 0))
    with col4:
        st.metric("Symptoms", summary.get("symptoms", 0))
    with col5:
        st.metric("Procedures", summary.get("procedures", 0))
    
    # Clinical summary
    if results.get("clinical_summary"):
        st.subheader("📋 Clinical Summary")
        st.info(results["clinical_summary"])
    
    # Display codes in organized sections
    codes = results.get("codes", [])
    if codes:
        st.subheader("🔍 Extracted ICD Codes")
        
        # Group codes by category
        categories = {}
        for code in codes:
            category = code.get("category", "Other")
            if category not in categories:
                categories[category] = []
            categories[category].append(code)
        
        # Display each category
        for category, category_codes in categories.items():
            with st.expander(f"📊 {category} ({len(category_codes)} codes)", expanded=True):
                for i, code in enumerate(category_codes):
                    # Create a card-like display for each code
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{code.get('code', 'N/A')}** - {code.get('description', 'No description')}")
                            
                            # Evidence
                            if code.get('evidence'):
                                st.markdown("*Evidence:*")
                                st.markdown(f"> {code['evidence']}")
                            
                            # Additional notes
                            if code.get('additional_notes'):
                                st.markdown(f"*Notes:* {code['additional_notes']}")
                        
                        with col2:
                            # Confidence badge
                            confidence = code.get('confidence', 'Unknown')
                            if confidence == 'High':
                                st.success(f"🟢 {confidence}")
                            elif confidence == 'Medium':
                                st.warning(f"🟡 {confidence}")
                            else:
                                st.error(f"🔴 {confidence}")
                    
                    if i < len(category_codes) - 1:
                        st.divider()
        
        # Create downloadable summary
        st.subheader("📄 Download Results")
        
        # Convert to DataFrame for better formatting
        df_data = []
        for code in codes:
            df_data.append({
                "ICD Code": code.get('code', ''),
                "Description": code.get('description', ''),
                "Category": code.get('category', ''),
                "Confidence": code.get('confidence', ''),
                "Evidence": code.get('evidence', ''),
                "Notes": code.get('additional_notes', '')
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Display as table
            st.dataframe(df, use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON download
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"icd_codes_{int(time.time())}.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV download
                csv_str = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_str,
                    file_name=f"icd_codes_{int(time.time())}.csv",
                    mime="text/csv"
                )
    
    # Coding notes
    if results.get("coding_notes"):
        st.subheader("📝 Coding Notes")
        st.warning(results["coding_notes"])

# Initialize client
client = get_gemini_client()
if not client:
    st.stop()

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["🎵 Audio Analysis", "📝 Text Analysis", "🔄 Combined Analysis"])

with tab1:
    st.subheader("Audio File ICD Code Extraction")
    st.write("Upload a medical audio file to automatically extract ICD-10/11 codes with evidence.")
    
    # Audio file uploader
    audio_file = st.file_uploader(
        "Choose a medical audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'],
        help="Supported formats: WAV, MP3, M4A, FLAC, OGG, WEBM (Max: 25MB per file)"
    )
    
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        
        # Show file info
        file_size = len(audio_file.getvalue()) / (1024 * 1024)  # Size in MB
        st.info(f"📁 **File**: {audio_file.name} | **Size**: {file_size:.2f} MB")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 Extract ICD Codes", type="primary"):
                with st.spinner("Analyzing audio for ICD codes... This may take a moment."):
                    # Reset file pointer
                    audio_file.seek(0)
                    
                    result = extract_icd_codes(
                        client, 
                        audio_file=audio_file, 
                        content_type="audio"
                    )
                    
                    if result:
                        st.session_state.icd_result = result
                        st.success("✅ ICD code extraction completed!")
        
        with col2:
            st.info("💡 **Tip**: Best results with clear audio containing medical consultations, diagnoses, or clinical discussions.")

with tab2:
    st.subheader("Text-Based ICD Code Extraction")
    st.write("Input medical text to extract relevant ICD-10/11 codes with supporting evidence.")
    
    text_input = st.text_area(
        "Enter medical text, clinical notes, or transcription:",
        height=200,
        placeholder="Enter patient notes, clinical documentation, discharge summaries, or any medical text containing diagnostic information..."
    )
    
    if text_input:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 Analyze Text", type="primary"):
                with st.spinner("Extracting ICD codes from text..."):
                    result = extract_icd_codes(
                        client,
                        text_content=text_input,
                        content_type="text"
                    )
                    
                    if result:
                        st.session_state.icd_result = result
                        st.success("✅ ICD code extraction completed!")
        
        with col2:
            st.info("💡 **Tip**: Include complete clinical information with symptoms, diagnoses, and treatment details for best results.")

with tab3:
    st.subheader("Combined Audio & Text Analysis")
    st.write("Combine audio input with additional text context for comprehensive ICD code extraction.")
    
    # Audio input
    combined_audio = st.file_uploader(
        "Medical audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'],
        key="combined_audio_icd"
    )
    
    # Text input
    combined_text = st.text_area(
        "Additional clinical context or notes:",
        height=150,
        placeholder="Add patient history, additional symptoms, or specific diagnostic questions...",
        key="combined_text_icd"
    )
    
    if combined_audio is not None or combined_text:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 Analyze Combined", type="primary"):
                with st.spinner("Processing combined input for ICD codes..."):
                    if combined_audio:
                        combined_audio.seek(0)
                    
                    result = extract_icd_codes(
                        client,
                        audio_file=combined_audio if combined_audio else None,
                        text_content=combined_text if combined_text else None,
                        content_type="both"
                    )
                    
                    if result:
                        st.session_state.icd_result = result
                        st.success("✅ ICD code extraction completed!")
        
        with col2:
            st.info("💡 **Tip**: Combine audio consultations with written patient history for the most comprehensive coding analysis.")

# Display results
if hasattr(st.session_state, 'icd_result') and st.session_state.icd_result:
    st.markdown("---")
    st.subheader("🎯 ICD Code Extraction Results")
    
    display_icd_results(st.session_state.icd_result)
    
    # Clear results button
    if st.button("🗑️ Clear Results"):
        if 'icd_result' in st.session_state:
            del st.session_state.icd_result
        st.rerun()

# Sidebar information
with st.sidebar:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **🔍 ICD-10/11 Extraction**: Automatic code identification
    - **📝 Evidence Tracking**: Direct quotes supporting each code
    - **🎯 Confidence Scoring**: High/Medium/Low confidence levels
    - **📊 Categorization**: Primary/Secondary diagnoses, symptoms, procedures
    - **💾 Export Options**: Download results as JSON or CSV
    """)
    
    st.markdown("### 📋 Code Categories")
    st.markdown("""
    **Primary Diagnoses**: Main conditions
    **Secondary Diagnoses**: Comorbidities
    **Symptoms**: Signs and symptoms
    **Procedures**: Medical interventions
    """)
    
    st.markdown("### 🏥 Use Cases")
    st.markdown("""
    - Medical coding assistance
    - Clinical documentation review
    - Billing and insurance support
    - Quality assurance
    - Coding education and training
    """)
    
    st.markdown("### ⚠️ Important Notice")
    st.warning("""
    This tool is for educational and assistance purposes only. 
    
    Always verify codes with qualified medical coding professionals before use in official documentation.
    """)
    
    st.markdown("### 📚 ICD Resources")
    st.markdown("""
    - [ICD-10-CM Official Guidelines](https://www.cdc.gov/nchs/icd/icd10cm.htm)
    - [WHO ICD-11](https://icd.who.int/browse11)
    - [CMS ICD-10 Resources](https://www.cms.gov/medicare/icd-10)
    """) 