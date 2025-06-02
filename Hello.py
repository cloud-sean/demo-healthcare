import streamlit as st

st.set_page_config(
    page_title="Medical AI Demos",
    page_icon="🏥",
    layout="wide"
)

st.write("# Welcome to Medical AI Demos! 🏥")

st.sidebar.success("Select a demo from the pages above.")

st.markdown(
    """
    This application showcases cutting-edge AI capabilities for healthcare applications 
    using Google's Gemini 2.5 Flash model. Explore various medical AI demonstrations 
    that can assist healthcare professionals in their daily workflows.
    
    **👈 Select a demo from the sidebar** to see what advanced AI can do in healthcare!
    
    ## Available Demos
    
    ### 🎤 Medical Transcription
    Transform audio recordings into structured medical notes with:
    - Audio file upload and real-time transcription
    - Speaker diarization for multi-participant conversations
    - Medical terminology understanding
    - Structured output formatting (including SOAP notes)
    - Summary generation
    
    ### 🔍 ICD Code Extraction
    Automatically extract ICD-10/11 codes from medical content with:
    - Audio and text analysis for diagnostic codes
    - Evidence-based code suggestions with supporting quotes
    - Confidence scoring (High/Medium/Low)
    - Category classification (Primary/Secondary diagnoses, symptoms, procedures)
    - Structured JSON output for integration
    - Export options (JSON/CSV) for further analysis
    
    ### 🧾 Invoice Processing
    Extract structured data from invoice images for healthcare billing:
    - Advanced image understanding for invoice data extraction
    - Complete vendor and billing information capture
    - Line item analysis with quantities and pricing
    - Financial summary with tax and total calculations
    - Payment terms and banking details extraction
    - Multiple export formats (JSON, CSV, summary reports)
    
    ### ✅ ICD-10 Code Selection
    Intelligent code selection from medical notes using checkbox interface:
    - Pre-selected list of common ICD-10 codes across all major categories
    - AI-powered analysis to determine which codes apply to specific medical notes
    - Evidence-based selections with supporting quotes from the medical documentation
    - Confidence scoring (High/Medium/Low) for each selected code
    - Primary/Secondary/Comorbidity categorization
    - Detailed clinical reasoning for code assignments
    - Export capabilities for selected codes and analysis results
    
    ## Features
    
    - **🎯 Medical-grade Accuracy**: Powered by Gemini 2.5 Flash with specialized medical knowledge
    - **🔊 Multi-modal Input**: Support for audio files, text input, images, or combination
    - **👥 Speaker Recognition**: Identify and separate different speakers in recordings
    - **📝 Structured Output**: Generate professional medical documentation
    - **⚡ Real-time Processing**: Fast transcription and analysis
    - **💾 Export Capabilities**: Download results in multiple formats
    - **🖼️ Image Understanding**: Advanced OCR and document analysis
    - **✅ Interactive Selection**: User-friendly checkbox interface for code selection
    
    ## Getting Started
    
    1. **Set up your API key**: Ensure you have a valid `GEMINI_API_KEY` in your environment
    2. **Choose a demo**: Select from the available demonstrations in the sidebar
    3. **Upload or input**: Provide your medical audio, text content, invoice images, or select ICD codes
    4. **Get results**: Receive professional-grade transcriptions, codes, data extraction, and code selections
    
    ## About the Technology
    
    These demos leverage Google's Gemini 2.5 Flash model, which excels at:
    - Understanding medical terminology and context
    - Processing long-form audio content (up to 8.4 hours)
    - Multi-modal analysis combining audio, text, and visual inputs
    - Generating human-like, contextually appropriate responses
    - Producing structured JSON output for integration
    - Advanced image understanding and document processing
    - Intelligent code selection with clinical reasoning
    
    ### Want to learn more?
    - Read about [Med-Gemini capabilities](https://research.google/blog/advancing-medical-ai-with-med-gemini/)
    - Explore [audio transcription with Gemini](https://cloud.google.com/blog/topics/partners/how-partners-unlock-scalable-audio-transcription-with-gemini)
    - Check out the [Gemini 2.5 Flash documentation](https://deepmind.google/models/gemini/flash/)
    - Learn about [Gemini image understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
    
    ---
    
    **⚠️ Important Note**: These demonstrations are for educational and research purposes. 
    Always consult with qualified healthcare professionals for medical decisions and verify 
    any extracted codes with certified medical coding specialists. Ensure data accuracy 
    for financial and billing applications. All ICD-10 code assignments should be validated 
    by qualified medical coding professionals.
    """
) 