import streamlit as st
import os
from google import genai
from google.genai import types
import time

st.set_page_config(page_title="Medical Transcription", page_icon="🎤", layout="centered", initial_sidebar_state="collapsed")

# Add Google Cloud logo using st.logo
st.logo("https://static-00.iconduck.com/assets.00/google-cloud-icon-2048x1646-7admxejz.png", size='large')

# Add MEDITECH logo centered
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="https://ehr.meditech.com/themes/ehrmeditech/images/meditech-logo.svg" style="height: 60px; width: auto;">
</div>
""", unsafe_allow_html=True)

# st.markdown("# 🎤 Medical Transcription")
# st.write("Upload audio files or record directly for professional medical transcription with speaker diarization.")

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



def transcript_only_prompt():
    """Return the transcript-only prompt."""
    return """You are a medical transcription specialist. Please transcribe the provided audio focusing ONLY on the conversation transcript:

1. **Speaker Diarization**: Identify and label different speakers (Doctor, Patient, Nurse, etc.)
2. **Accuracy**: Capture the exact conversation with proper medical terminology
3. **Format**: Present as a clean conversation transcript with speaker labels

Format the output as:
- **Participants**: List all speakers identified
- **Transcript**: Complete conversation with clear speaker labels like [Doctor], [Patient], [Nurse], etc.

Do NOT include medical summaries, diagnoses, or action items - focus only on transcribing what was said."""

def medical_summary_prompt():
    """Return the medical summary/SOAP note prompt."""
    return """Based on the medical transcription, provide ONLY the medical analysis and summary in SOAP note format:

**SOAP NOTE:**

**Subjective:**
- Chief complaint and patient's description of symptoms
- Relevant medical history mentioned

**Objective:**
- Physical examination findings mentioned
- Vital signs or test results discussed

**Assessment:**
- Primary diagnosis or working diagnosis
- Differential diagnoses considered

**Plan:**
- Treatment recommendations
- Medications prescribed
- Follow-up instructions
- Referrals or additional tests

Do NOT include the conversation transcript - focus only on the clinical analysis and medical decision-making."""

def brief_summary_prompt():
    """Return the brief summary prompt."""
    return """Based on the medical transcription, provide a concise brief summary with only the most essential information:

- **Chief Complaint**: Main reason for visit (1-2 sentences)
- **Key Findings**: Most important medical findings
- **Primary Diagnosis**: Main diagnosis or suspected condition
- **Treatment Plan**: Key treatments or medications prescribed
- **Follow-up**: Essential next steps

Keep this summary under 200 words and focus only on critical medical information."""

def extended_summary_prompt():
    """Return the extended summary prompt for internal medicine."""
    return """Based on the medical transcription, provide a comprehensive extended summary specifically tailored for an internal medicine practitioner:

**EXTENDED CLINICAL SUMMARY:**

**Chief Complaint & History of Present Illness:**
- Detailed patient presentation and symptom timeline
- Relevant associated symptoms and clinical context

**Past Medical History & Review of Systems:**
- Significant medical history, medications, allergies
- Relevant family history and social history

**Clinical Assessment:**
- Detailed physical examination findings
- Vital signs and any diagnostic test results mentioned

**Differential Diagnosis:**
- Primary working diagnosis with clinical reasoning
- Alternative diagnoses considered and why
- Risk stratification if applicable

**Clinical Decision Making:**
- Rationale for chosen diagnostic approach
- Treatment selection reasoning
- Consideration of comorbidities and drug interactions

**Management Plan:**
- Detailed treatment recommendations
- Medication dosing and monitoring requirements
- Lifestyle modifications and patient education
- Follow-up schedule and red flag symptoms

**Clinical Pearls & Considerations:**
- Internal medicine insights relevant to the case
- Potential complications to monitor
- Interdisciplinary care considerations

Provide a comprehensive analysis suitable for medical education and clinical decision-making."""

def setswana_summary_prompt():
    """Return the Setswana summary translation prompt."""
    return """Translate the following brief medical summary into Setswana language while maintaining medical accuracy:

Requirements:
- Translate the brief summary content only
- Preserve medical terminology where appropriate Setswana equivalents exist
- Keep medical terms in English where no clear Setswana equivalent exists (followed by Setswana explanation in parentheses)
- Maintain clear, simple structure suitable for patient communication
- Ensure cultural sensitivity and appropriate medical communication style for Setswana speakers
- Keep the same brevity and focus as the original summary

Provide a concise Setswana translation that would be useful for Setswana-speaking patients."""

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

def generate_medical_transcription(client, audio_file):
    """Generate medical transcription using Gemini."""
    try:
        model = "gemini-2.5-flash-preview-05-20"
        
        # Read audio file and create proper audio part
        audio_bytes = audio_file.read()
        mime_type = get_audio_mime_type(audio_file.name)
        
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="text/plain",
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        usage_metadata = None
        
        # Generate transcript only (base transcription)
        status_text.text("Generating transcript...")
        
        transcript_parts = [
            types.Part.from_text(text=transcript_only_prompt()),
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        ]
        transcript_contents = [types.Content(role="user", parts=transcript_parts)]
        
        transcript_only = ""
        chunks_received = 0
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=transcript_contents,
            config=generate_content_config,
        ):
            if chunk.text:
                transcript_only += chunk.text
                chunks_received += 1
                
                # Update progress (estimated)
                progress = min(chunks_received * 0.05, 0.25)
                progress_bar.progress(progress)
                status_text.text(f"Transcript... {len(transcript_only)} characters")
            
            # Capture usage metadata from the chunk if available
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata
        
        # Generate medical summary/SOAP note
        progress_bar.progress(0.33)
        status_text.text("Generating SOAP note...")
        
        medical_parts = [
            types.Part.from_text(text=f"{medical_summary_prompt()}\n\nTranscript:\n{transcript_only}")
        ]
        medical_contents = [types.Content(role="user", parts=medical_parts)]
        
        medical_summary = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=medical_contents,
            config=generate_content_config,
        ):
            if chunk.text:
                medical_summary += chunk.text
        
        # Generate brief summary
        progress_bar.progress(0.5)
        status_text.text("Generating brief summary...")
        
        brief_parts = [
            types.Part.from_text(text=f"{brief_summary_prompt()}\n\nTranscript:\n{transcript_only}")
        ]
        brief_contents = [types.Content(role="user", parts=brief_parts)]
        
        brief_summary = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=brief_contents,
            config=generate_content_config,
        ):
            if chunk.text:
                brief_summary += chunk.text
        
        # Generate extended summary
        progress_bar.progress(0.66)
        status_text.text("Generating extended summary...")
        
        extended_parts = [
            types.Part.from_text(text=f"{extended_summary_prompt()}\n\nTranscript:\n{transcript_only}")
        ]
        extended_contents = [types.Content(role="user", parts=extended_parts)]
        
        extended_summary = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=extended_contents,
            config=generate_content_config,
        ):
            if chunk.text:
                extended_summary += chunk.text
        
        # Generate Setswana summary (only brief summary translation)
        progress_bar.progress(0.85)
        status_text.text("Generating Setswana summary...")
        
        setswana_parts = [
            types.Part.from_text(text=f"{setswana_summary_prompt()}\n\nBrief summary to translate:\n{brief_summary}")
        ]
        setswana_contents = [types.Content(role="user", parts=setswana_parts)]
        
        setswana_summary = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=setswana_contents,
            config=generate_content_config,
        ):
            if chunk.text:
                setswana_summary += chunk.text
        
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return {
            'transcript': transcript_only,
            'medical_summary': medical_summary,
            'brief': brief_summary,
            'extended': extended_summary,
            'setswana': setswana_summary
        }, usage_metadata
        
    except Exception as e:
        st.error(f"Error generating transcription: {str(e)}")
        return None, None

# Initialize client
client = get_gemini_client()
if not client:
    st.stop()

# Main interface with tabs
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])

with tab1:
    st.markdown("#### Upload Audio File")
    audio_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'],
        help="Supported: WAV, MP3, M4A, FLAC, OGG, WEBM (Max: 25MB)"
    )
    
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        file_size = len(audio_file.getvalue()) / (1024 * 1024)
        st.info(f"📁 {audio_file.name} | {file_size:.2f} MB")
        
        if st.button("🎤 Transcribe Audio", type="primary", key="upload_transcribe"):
            with st.spinner("Transcribing audio..."):
                audio_file.seek(0)
                results, usage_metadata = generate_medical_transcription(client, audio_file)
                
                if results:
                    st.session_state.transcription_results = results
                    st.session_state.usage_metadata = usage_metadata
                    st.success("✅ All transcriptions completed!")

with tab2:
    st.markdown("#### Record Audio")
    recorded_audio = st.audio_input("Record a medical conversation")
    
    if recorded_audio is not None:
        st.audio(recorded_audio, format='audio/wav')
        recording_size = len(recorded_audio.getvalue()) / (1024 * 1024)
        st.info(f"🎙️ Recording | {recording_size:.2f} MB")
        
        if st.button("🎤 Transcribe Recording", type="primary", key="record_transcribe"):
            with st.spinner("Transcribing recording..."):
                recorded_audio.seek(0)
                results, usage_metadata = generate_medical_transcription(client, recorded_audio)
                
                if results:
                    st.session_state.transcription_results = results
                    st.session_state.usage_metadata = usage_metadata
                    st.success("✅ All transcriptions completed!")

# Display results
if hasattr(st.session_state, 'transcription_results') and st.session_state.transcription_results:
    st.markdown("---")
    st.subheader("📋 Medical Transcription Results")
    
    # Display token usage information if available
    if hasattr(st.session_state, 'usage_metadata') and st.session_state.usage_metadata:
        usage = st.session_state.usage_metadata
        st.info(f"📊 **Tokens Used**: {usage.total_token_count:,} (Input: {usage.prompt_token_count:,}, Output: {usage.candidates_token_count:,})")
    
    # Create tabs for different result versions
    result_tab1, result_tab2, result_tab3, result_tab4, result_tab5 = st.tabs([
        "⚡ Brief Summary",
        "📋 Extended Summary",
        "🏥 SOAP Note", 
        "🌍 Setswana Summary",
        "💬 Transcript"
    ])
    
    with result_tab1:
        st.markdown(st.session_state.transcription_results['brief'])
        
        st.download_button(
            label="📄 Download Brief Summary",
            data=st.session_state.transcription_results['brief'],
            file_name=f"brief_summary_{int(time.time())}.txt",
            mime="text/plain"
        )
    
    with result_tab2:
        st.markdown(st.session_state.transcription_results['extended'])
        
        st.download_button(
            label="📄 Download Extended Summary",
            data=st.session_state.transcription_results['extended'],
            file_name=f"extended_summary_{int(time.time())}.txt",
            mime="text/plain"
        )
    
    with result_tab3:
        st.markdown(st.session_state.transcription_results['medical_summary'])
        
        st.download_button(
            label="📄 Download SOAP Note",
            data=st.session_state.transcription_results['medical_summary'],
            file_name=f"soap_note_{int(time.time())}.txt",
            mime="text/plain"
        )
    
    with result_tab4:
        st.markdown(st.session_state.transcription_results['setswana'])
        
        st.download_button(
            label="📄 Download Setswana Summary",
            data=st.session_state.transcription_results['setswana'],
            file_name=f"setswana_summary_{int(time.time())}.txt",
            mime="text/plain"
        )
    
    with result_tab5:
        st.markdown(st.session_state.transcription_results['transcript'])
        
        st.download_button(
            label="📄 Download Transcript",
            data=st.session_state.transcription_results['transcript'],
            file_name=f"transcript_{int(time.time())}.txt",
            mime="text/plain"
        )
    
    # Clear results button
    if st.button("🗑️ Clear All Results"):
        if 'transcription_results' in st.session_state:
            del st.session_state.transcription_results
        if 'usage_metadata' in st.session_state:
            del st.session_state.usage_metadata
        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - Audio transcription with speaker diarization
    - Clean conversation transcript
    - Professional SOAP note generation
    - Brief summary for quick reference
    - Extended summary for internal medicine
    - Setswana summary for patient communication
    - Fast processing with Gemini 2.5 Flash
    """)
    
    st.markdown("### ⚠️ Privacy Notice")
    st.warning("This is a demo. For production use, ensure HIPAA compliance and secure connections.") 