import streamlit as st
import base64
import os
import io
from google import genai
from google.genai import types
import time

st.set_page_config(page_title="Medical Transcription", page_icon="🎤", layout="wide")

st.markdown("# 🎤 Medical Transcription")
st.sidebar.header("Medical Transcription Demo")

st.write(
    """
    This demo showcases advanced medical transcription capabilities using Gemini 2.5 Flash. 
    Upload audio files or input text to generate professional medical documentation with 
    speaker diarization, medical terminology understanding, and structured formatting.
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

def medical_transcription_prompt():
    """Return the medical transcription prompt."""
    return """You are a medical transcription specialist. Please transcribe the provided audio with the following requirements:

1. **Medical Accuracy**: Use proper medical terminology and maintain clinical precision
2. **Speaker Diarization**: Identify and label different speakers (Doctor, Patient, Nurse, etc.)
3. **Structure**: Format as a professional medical document with appropriate sections
4. **Completeness**: Include all medical information, symptoms, diagnoses, and treatment plans
5. **Timestamps**: Add relevant timestamps for key sections if possible

Format the output as:
- **Participants**: List all speakers identified
- **Transcription**: Complete conversation with speaker labels
- **Medical Summary**: Key medical points, diagnoses, and recommendations - SOAP note format
- **Action Items**: Any follow-up actions or prescriptions mentioned

If the audio contains multiple speakers, clearly distinguish between them using labels like [Doctor], [Patient], [Nurse], etc."""

def text_analysis_prompt():
    """Return the text analysis prompt for medical content."""
    return """You are a medical documentation specialist. Please analyze the provided medical text and enhance it with the following:

1. **Medical Accuracy**: Verify and enhance medical terminology
2. **Structure**: Organize into professional medical documentation format
3. **Completeness**: Identify any missing critical information
4. **Clarity**: Improve readability while maintaining medical precision
5. **Compliance**: Ensure format meets medical documentation standards

Please provide:
- **Enhanced Medical Text**: Professionally formatted version
- **Medical Summary**: Key medical points and findings
- **Recommendations**: Suggested improvements or missing information
- **Terminology Check**: Any medical terms that need clarification or correction"""

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

def generate_medical_transcription(client, audio_file=None, text_content=None, content_type="audio"):
    """Generate medical transcription using Gemini."""
    try:
        model = "gemini-2.5-flash-preview-05-20"
        
        # Prepare content based on input type
        parts = []
        
        if content_type == "audio" and audio_file:
            parts.append(types.Part.from_text(text=medical_transcription_prompt()))
            
            # Read audio file and create proper audio part
            audio_bytes = audio_file.read()
            mime_type = get_audio_mime_type(audio_file.name)
            
            # Create audio part for Gemini
            parts.append(types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type
            ))
            
        elif content_type == "text" and text_content:
            parts.append(types.Part.from_text(text=text_analysis_prompt()))
            parts.append(types.Part.from_text(text=f"Medical text to analyze:\n\n{text_content}"))
            
        elif content_type == "both":
            parts.append(types.Part.from_text(text=medical_transcription_prompt()))
            
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
            response_mime_type="text/plain",
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
                progress = min(chunks_received * 0.1, 0.9)
                progress_bar.progress(progress)
                status_text.text(f"Processing... {len(response_text)} characters generated")
        
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        time.sleep(0.5)  # Brief pause to show completion
        progress_bar.empty()
        status_text.empty()
                
        return response_text
        
    except Exception as e:
        st.error(f"Error generating transcription: {str(e)}")
        return None

# Initialize client
client = get_gemini_client()
if not client:
    st.stop()

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["🎵 Audio Upload", "📝 Text Input", "🔄 Combined Input"])

with tab1:
    st.subheader("Audio File Transcription")
    st.write("Upload an audio file for medical transcription with speaker diarization.")
    
    # Audio file uploader
    audio_file = st.file_uploader(
        "Choose an audio file",
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
            if st.button("🎤 Transcribe Audio", type="primary"):
                with st.spinner("Transcribing audio... This may take a moment."):
                    # Reset file pointer
                    audio_file.seek(0)
                    
                    result = generate_medical_transcription(
                        client, 
                        audio_file=audio_file, 
                        content_type="audio"
                    )
                    
                    if result:
                        st.session_state.transcription_result = result
                        st.success("✅ Transcription completed!")
        
        with col2:
            st.info("💡 **Tip**: For best results, ensure clear audio with minimal background noise.")

with tab2:
    st.subheader("Text Analysis & Enhancement")
    st.write("Input medical text for analysis, formatting, and enhancement.")
    
    text_input = st.text_area(
        "Enter medical text or notes:",
        height=200,
        placeholder="Enter medical notes, patient conversations, or clinical documentation here..."
    )
    
    if text_input:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("📝 Analyze Text", type="primary"):
                with st.spinner("Analyzing medical text..."):
                    result = generate_medical_transcription(
                        client,
                        text_content=text_input,
                        content_type="text"
                    )
                    
                    if result:
                        st.session_state.transcription_result = result
                        st.success("✅ Analysis completed!")
        
        with col2:
            st.info("💡 **Tip**: Include patient symptoms, medical history, and any clinical observations.")

with tab3:
    st.subheader("Combined Audio & Text Processing")
    st.write("Upload audio and provide additional text context for comprehensive analysis.")
    
    # Audio input
    combined_audio = st.file_uploader(
        "Audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'],
        key="combined_audio"
    )
    
    # Text input
    combined_text = st.text_area(
        "Additional context or notes:",
        height=150,
        placeholder="Add any additional context, patient history, or specific questions...",
        key="combined_text"
    )
    
    if combined_audio is not None or combined_text:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔄 Process Combined", type="primary"):
                with st.spinner("Processing combined input..."):
                    if combined_audio:
                        combined_audio.seek(0)
                    
                    result = generate_medical_transcription(
                        client,
                        audio_file=combined_audio if combined_audio else None,
                        text_content=combined_text if combined_text else None,
                        content_type="both"
                    )
                    
                    if result:
                        st.session_state.transcription_result = result
                        st.success("✅ Processing completed!")
        
        with col2:
            st.info("💡 **Tip**: Combine audio recordings with written notes for the most comprehensive analysis.")

# Display results
if hasattr(st.session_state, 'transcription_result') and st.session_state.transcription_result:
    st.markdown("---")
    st.subheader("📋 Medical Transcription Results")
    
    # Create expandable sections for better organization
    with st.expander("📝 Full Transcription", expanded=True):
        st.markdown(st.session_state.transcription_result)
    
    # Option to download results
    st.download_button(
        label="📄 Download Transcription",
        data=st.session_state.transcription_result,
        file_name=f"medical_transcription_{int(time.time())}.txt",
        mime="text/plain"
    )
    
    # Clear results button
    if st.button("🗑️ Clear Results"):
        if 'transcription_result' in st.session_state:
            del st.session_state.transcription_result
        st.rerun()

# Sidebar information
with st.sidebar:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **🎤 Audio Transcription**: Upload medical audio files
    - **👥 Speaker Diarization**: Identify different speakers
    - **📝 Text Enhancement**: Improve medical documentation
    - **🔄 Multi-modal**: Combine audio and text inputs
    - **⚡ Real-time**: Fast processing with Gemini 2.5 Flash
    """)
    
    st.markdown("### 📊 Supported Formats")
    st.markdown("""
    **Audio**: WAV, MP3, M4A, FLAC, OGG, WEBM
    
    **Content Types**:
    - Patient consultations
    - Medical dictations
    - Clinical meetings
    - Procedure notes
    - Treatment discussions
    """)
    
    st.markdown("### ⚠️ Privacy Notice")
    st.warning("""
    This is a demonstration. In production:
    - Ensure HIPAA compliance
    - Use secure, encrypted connections
    - Implement proper access controls
    - Follow medical data regulations
    """)
    
    st.markdown("### 🛠️ Technical Notes")
    st.info("""
    - Max file size: 25MB
    - Supported audio length: Up to ~8.4 hours
    - Model: Gemini 2.5 Flash
    - Processing: Real-time streaming
    """) 