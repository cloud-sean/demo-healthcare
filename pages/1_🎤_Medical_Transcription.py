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
        usage_metadata = None
        
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
            
            # Capture usage metadata from the chunk if available
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage_metadata = chunk.usage_metadata
        
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        time.sleep(0.5)  # Brief pause to show completion
        progress_bar.empty()
        status_text.empty()
        
        return response_text, usage_metadata
        
    except Exception as e:
        st.error(f"Error generating transcription: {str(e)}")
        return None, None

# Initialize client
client = get_gemini_client()
if not client:
    st.stop()

# Create tabs for different input methods
tab1, tab2, tab3, tab4 = st.tabs(["🎵 Audio Upload", "🎙️ Voice Recording", "📝 Text Input", "🔄 Combined Input"])

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
                    
                    result, usage_metadata = generate_medical_transcription(
                        client, 
                        audio_file=audio_file, 
                        content_type="audio"
                    )
                    
                    if result:
                        st.session_state.transcription_result = result
                        st.session_state.usage_metadata = usage_metadata
                        st.success("✅ Transcription completed!")
        
        with col2:
            st.info("💡 **Tip**: For best results, ensure clear audio with minimal background noise.")

with tab2:
    st.subheader("🎙️ Live Voice Recording")
    st.write("Record conversations, dictations, or medical notes directly in your browser.")
    
    # Voice recording interface with HTML/JavaScript
    st.markdown("""
    <div id="voice-recorder" style="padding: 20px; border: 2px dashed #cccccc; border-radius: 10px; text-align: center;">
        <h4>🎤 Medical Voice Recorder</h4>
        <p id="recording-status">Click 'Start Recording' to begin</p>
        <div style="margin: 20px 0;">
            <button id="start-btn" onclick="startRecording()" style="background-color: #ff4b4b; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 0 10px; cursor: pointer;">
                🔴 Start Recording
            </button>
            <button id="stop-btn" onclick="stopRecording()" style="background-color: #262730; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 0 10px; cursor: pointer;" disabled>
                ⏹️ Stop Recording
            </button>
            <button id="clear-btn" onclick="clearRecording()" style="background-color: #666; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 0 10px; cursor: pointer;" disabled>
                🗑️ Clear
            </button>
        </div>
        <div id="recording-timer" style="font-size: 24px; font-weight: bold; color: #ff4b4b; margin: 10px 0; display: none;">
            00:00
        </div>
        <audio id="recorded-audio" controls style="width: 100%; margin: 20px 0; display: none;"></audio>
        <div id="download-section" style="margin: 20px 0; display: none;">
            <button id="transcribe-btn" onclick="transcribeRecording()" style="background-color: #00cc88; color: white; border: none; padding: 12px 24px; border-radius: 5px; margin: 0 10px; cursor: pointer; font-weight: bold;">
                🎤➡️📝 Transcribe Recording
            </button>
        </div>
    </div>
    
    <script>
    let mediaRecorder;
    let recordedChunks = [];
    let isRecording = false;
    let recordingTimer;
    let recordingStartTime;
    let recordedBlob = null;
    
    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100,
                    channelCount: 1
                } 
            });
            
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            recordedChunks = [];
            
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = function() {
                recordedBlob = new Blob(recordedChunks, { type: 'audio/webm' });
                const audioUrl = URL.createObjectURL(recordedBlob);
                
                const audioElement = document.getElementById('recorded-audio');
                audioElement.src = audioUrl;
                audioElement.style.display = 'block';
                
                document.getElementById('download-section').style.display = 'block';
                document.getElementById('clear-btn').disabled = false;
                
                // Stop all tracks to release microphone
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start(1000); // Collect data every second
            isRecording = true;
            recordingStartTime = Date.now();
            
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('recording-status').textContent = '🔴 Recording in progress...';
            document.getElementById('recording-timer').style.display = 'block';
            
            // Start timer
            recordingTimer = setInterval(updateTimer, 1000);
            
        } catch (error) {
            console.error('Error accessing microphone:', error);
            document.getElementById('recording-status').textContent = '❌ Error: Could not access microphone. Please allow microphone access and try again.';
        }
    }
    
    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
            document.getElementById('recording-status').textContent = '✅ Recording completed! You can now play it back or transcribe it.';
            document.getElementById('recording-timer').style.display = 'none';
            
            clearInterval(recordingTimer);
        }
    }
    
    function clearRecording() {
        const audioElement = document.getElementById('recorded-audio');
        audioElement.src = '';
        audioElement.style.display = 'none';
        
        document.getElementById('download-section').style.display = 'none';
        document.getElementById('clear-btn').disabled = true;
        document.getElementById('recording-status').textContent = 'Click \\'Start Recording\\' to begin';
        document.getElementById('recording-timer').textContent = '00:00';
        
        recordedChunks = [];
        recordedBlob = null;
    }
    
    function updateTimer() {
        if (isRecording) {
            const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            document.getElementById('recording-timer').textContent = 
                String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
        }
    }
    
    function transcribeRecording() {
        if (recordedBlob) {
            // Create a File object from the blob
            const audioFile = new File([recordedBlob], 'recorded_audio.webm', { type: 'audio/webm' });
            
            // Store the audio file in a way that Streamlit can access
            // We'll use a custom event to communicate with Streamlit
            const event = new CustomEvent('audioRecorded', {
                detail: { audioBlob: recordedBlob }
            });
            
            // Set a flag that Streamlit can check
            window.recordedAudioReady = true;
            window.recordedAudioBlob = recordedBlob;
            
            document.getElementById('recording-status').textContent = '🔄 Preparing audio for transcription...';
            
            // Trigger Streamlit rerun by clicking a hidden button
            setTimeout(() => {
                if (window.streamlitRerun) {
                    window.streamlitRerun();
                }
            }, 100);
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Alternative approach: Manual file upload after recording
    st.markdown("### 📤 Upload Your Recording")
    st.write("After recording above, save your audio and upload it here for transcription:")
    
    recorded_audio_file = st.file_uploader(
        "Upload recorded audio",
        type=['webm', 'wav', 'mp3', 'm4a', 'ogg'],
        help="Record using the interface above, then save and upload the file here",
        key="recorded_audio_upload"
    )
    
    if recorded_audio_file is not None:
        st.audio(recorded_audio_file, format='audio/wav')
        
        # Show file info
        file_size = len(recorded_audio_file.getvalue()) / (1024 * 1024)  # Size in MB
        st.info(f"📁 **File**: {recorded_audio_file.name} | **Size**: {file_size:.2f} MB")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🎤 Transcribe Recording", type="primary", key="transcribe_recording"):
                with st.spinner("Transcribing your recording... This may take a moment."):
                    # Reset file pointer
                    recorded_audio_file.seek(0)
                    
                    result, usage_metadata = generate_medical_transcription(
                        client, 
                        audio_file=recorded_audio_file, 
                        content_type="audio"
                    )
                    
                    if result:
                        st.session_state.transcription_result = result
                        st.session_state.usage_metadata = usage_metadata
                        st.success("✅ Recording transcribed successfully!")
        
        with col2:
            st.info("💡 **Tip**: Your recording is ready for transcription. Click the button to process it.")
    
    # Instructions and tips
    st.markdown("### 💡 Recording Tips:")
    st.info("""
    - **🎧 Use headphones** to prevent audio feedback
    - **🔇 Find a quiet environment** for better transcription accuracy
    - **🗣️ Speak clearly** and at a moderate pace
    - **📏 Keep recordings under 10 minutes** for optimal processing
    - **🔊 Check your microphone levels** before important recordings
    """)
    
    st.markdown("### 🔒 Privacy & Security:")
    st.warning("""
    - All recordings are processed locally in your browser
    - Audio data is only sent to Gemini API for transcription
    - No recordings are permanently stored on our servers
    - For production use, ensure HIPAA compliance measures
    """)

with tab3:
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
                    result, usage_metadata = generate_medical_transcription(
                        client,
                        text_content=text_input,
                        content_type="text"
                    )
                    
                    if result:
                        st.session_state.transcription_result = result
                        st.session_state.usage_metadata = usage_metadata
                        st.success("✅ Analysis completed!")
        
        with col2:
            st.info("💡 **Tip**: Include patient symptoms, medical history, and any clinical observations.")

with tab4:
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
                    
                    result, usage_metadata = generate_medical_transcription(
                        client,
                        audio_file=combined_audio if combined_audio else None,
                        text_content=combined_text if combined_text else None,
                        content_type="both"
                    )
                    
                    if result:
                        st.session_state.transcription_result = result
                        st.session_state.usage_metadata = usage_metadata
                        st.success("✅ Processing completed!")
        
        with col2:
            st.info("💡 **Tip**: Combine audio recordings with written notes for the most comprehensive analysis.")

# Display results
if hasattr(st.session_state, 'transcription_result') and st.session_state.transcription_result:
    st.markdown("---")
    st.subheader("📋 Medical Transcription Results")
    
    # Display token usage information if available
    if hasattr(st.session_state, 'usage_metadata') and st.session_state.usage_metadata:
        usage = st.session_state.usage_metadata
        st.info(f"""
        📊 **Token Usage:**
        - **Input tokens**: {usage.prompt_token_count:,}
        - **Output tokens**: {usage.candidates_token_count:,}
        - **Total tokens**: {usage.total_token_count:,}
        """)
    
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
        if 'usage_metadata' in st.session_state:
            del st.session_state.usage_metadata
        st.rerun()

# Sidebar information
with st.sidebar:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **🎤 Audio Transcription**: Upload medical audio files
    - **🎙️ Live Recording**: Record conversations directly in browser
    - **👥 Speaker Diarization**: Identify different speakers
    - **📝 Text Enhancement**: Improve medical documentation
    - **🔄 Multi-modal**: Combine audio and text inputs
    - **⚡ Real-time**: Fast processing with Gemini 2.5 Flash
    - **📊 Token Tracking**: Monitor API usage and costs
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