# Medical AI Demos with Gemini 2.5 Flash 🏥

A multipage Streamlit application showcasing cutting-edge AI capabilities for healthcare applications using Google's Gemini 2.5 Flash model.

## Features

### 🎤 Medical Transcription Demo
- **Audio File Upload**: Support for WAV, MP3, M4A, FLAC, OGG, WEBM formats
- **Speaker Diarization**: Automatically identify and separate different speakers
- **Medical Terminology**: Specialized understanding of medical language and context
- **Multi-modal Input**: Process audio files, text input, or combination of both
- **Structured Output**: Generate professional medical documentation
- **Real-time Processing**: Stream responses with progress indicators

### 🔍 ICD Code Extraction Demo
- **Automatic Code Detection**: Extract ICD-10/11 codes from medical audio or text
- **Evidence-Based Results**: Each code includes supporting quotes from the source
- **Confidence Scoring**: High/Medium/Low confidence levels for each extraction
- **Smart Categorization**: Classify codes as Primary/Secondary diagnoses, symptoms, procedures
- **JSON Output**: Structured data format for easy integration
- **Export Options**: Download results as JSON or CSV files

### 🧾 Invoice Processing Demo
- **Image Understanding**: Advanced OCR and document analysis for invoices
- **Complete Data Extraction**: Vendor info, billing details, line items, payment terms
- **Financial Analysis**: Automatic calculation of totals, taxes, and discounts
- **Structured JSON Output**: Ready for integration with accounting systems
- **Multiple Export Options**: JSON, CSV, and summary report downloads
- **Invoice Type Detection**: Standard invoices, receipts, credit notes, purchase orders

### ✅ ICD-10 Code Selection Demo
- **Interactive Code Selection**: Checkbox interface with 80+ common ICD-10 codes
- **Smart AI Analysis**: Determine which codes apply to specific medical notes
- **Evidence-Based Results**: Supporting quotes from medical documentation for each code
- **Confidence Scoring**: High/Medium/Low confidence levels with clinical reasoning
- **Code Categorization**: Primary/Secondary/Comorbidity classification
- **Comprehensive Coverage**: Codes across 10 major medical categories
- **Export Options**: JSON, CSV, and summary report downloads

### 📝 Paper to Patient Note Demo
- **Advanced Handwriting Recognition**: AI-powered OCR specialized for medical handwriting
- **Medical Terminology Focus**: Accurate extraction of medical terms, drug names, and abbreviations
- **Multiple Output Formats**: Structured JSON analysis or clean text extraction
- **Quality Assessment**: Legibility scoring and confidence evaluation for extracted text
- **Structure Preservation**: Maintain original formatting and identify note sections
- **Correction Tracking**: Identify and note crossed-out or corrected text
- **Digital Integration**: Export options for EHR systems and digital workflows

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key

## Installation

1. **Clone or download the repository**
   ```bash
   cd your-project-directory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Gemini API key**
   
   You need to obtain a Gemini API key from Google Cloud. Once you have it, set it as an environment variable:
   
   ```bash
   # On macOS/Linux:
   export GEMINI_API_KEY="your_api_key_here"
   
   # On Windows:
   set GEMINI_API_KEY=your_api_key_here
   ```
   
   Alternatively, you can create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Running the Application

Start the Streamlit app:

```bash
streamlit run Hello.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage Guide

### Medical Transcription Demo

1. **Navigate to the Medical Transcription page** using the sidebar
2. **Choose your input method**:
   - **Audio Upload**: Upload medical audio files for transcription
   - **Text Input**: Enter medical text for analysis and enhancement
   - **Combined Input**: Use both audio and text for comprehensive analysis

3. **Upload/Input your content**:
   - For audio: Choose supported formats (WAV, MP3, M4A, FLAC, OGG, WEBM)
   - For text: Enter medical notes, conversations, or documentation
   - Maximum file size: 25MB
   - Maximum audio length: ~8.4 hours

4. **Process and review results**:
   - View structured transcription with speaker labels
   - Download results as text files
   - Clear results when done

### ICD Code Extraction Demo

1. **Navigate to the ICD Code Extraction page** using the sidebar
2. **Choose your input method**:
   - **Audio Analysis**: Upload medical audio files for code extraction
   - **Text Analysis**: Enter medical text for code identification
   - **Combined Analysis**: Use both audio and text for comprehensive analysis

3. **Upload/Input your content**:
   - Medical audio files containing diagnostic information
   - Clinical notes, discharge summaries, or medical documentation
   - Patient consultation recordings

4. **Review extracted codes**:
   - View summary metrics (total codes, categories)
   - Examine detailed code information with evidence
   - Check confidence levels for each extraction
   - Download results as JSON or CSV

### Invoice Processing Demo

1. **Navigate to the Invoice Processing page** using the sidebar
2. **Upload an invoice image**:
   - Supported formats: PNG, JPG, JPEG, WEBP, HEIC, HEIF
   - Maximum file size: 20MB
   - Ensure clear, readable images for best results

3. **Process and review results**:
   - View extraction confidence and invoice metadata
   - Review detailed vendor and billing information
   - Examine line items and financial summaries
   - Check payment terms and banking details
   - Download structured data as JSON, CSV, or summary report

### ICD-10 Code Selection Demo

1. **Navigate to the ICD Code Selection page** using the sidebar
2. **Select potential ICD-10 codes**:
   - Browse 10 medical categories with 80+ common codes
   - Use checkboxes to select codes that might apply to your case
   - Use "Select All" / "Deselect All" buttons for entire categories
   - Categories include: Cardiovascular, Endocrine, Mental Health, Respiratory, etc.

3. **Enter your medical note**:
   - Provide complete clinical documentation
   - Include chief complaint, history, physical exam, and assessment
   - Use standard medical terminology for best results

4. **Review AI analysis results**:
   - View summary metrics and confidence distribution
   - Examine selected codes with supporting evidence
   - Review clinical reasoning for each code assignment
   - Check primary vs. secondary diagnosis categorization
   - Download analysis as JSON, CSV, or summary report

### Paper to Patient Note Demo

1. **Navigate to the Paper to Patient Note page** using the sidebar
2. **Upload handwritten note image**:
   - Supported formats: PNG, JPG, JPEG, WEBP, HEIC, HEIF
   - Maximum file size: 20MB
   - Ensure clear, well-lit images for best OCR results
   - Include full page boundaries in the image

3. **Configure conversion options**:
   - Choose output format: Structured JSON analysis or Raw text
   - Select structure preservation: Maintain original layout vs. clean formatting
   - Review image preview and file information

4. **Review conversion results**:
   - View extracted text with quality assessment metrics
   - Check legibility scores and confidence levels
   - Examine identified sections and medical terminology
   - Review annotations, corrections, and formatting suggestions
   - Download extracted text, analysis, or summary reports

## Features Showcase

### Audio Processing Capabilities
- **High Accuracy**: Leverages Gemini's advanced speech recognition
- **Speaker Diarization**: Identifies Doctor, Patient, Nurse, etc.
- **Medical Context**: Understands medical terminology and procedures
- **Long-form Support**: Process up to 8.4 hours of audio content

### Text Analysis Features
- **Medical Enhancement**: Improve documentation format and structure
- **Terminology Validation**: Check and correct medical terms
- **Compliance**: Ensure medical documentation standards
- **Summary Generation**: Extract key medical points and recommendations

### ICD Code Extraction Features
- **Intelligent Detection**: Automatically identify relevant diagnostic codes
- **Evidence Tracking**: Link each code to supporting text evidence
- **Category Classification**: Organize codes by type (Primary/Secondary/Symptoms/Procedures)
- **Confidence Assessment**: Evaluate the reliability of each extraction
- **Structured Output**: JSON format for easy integration with other systems

### Image Understanding Features
- **Advanced OCR**: Extract text from various invoice formats and layouts
- **Document Analysis**: Understand invoice structure and relationships
- **Data Validation**: Verify numerical calculations and consistency
- **Multi-language Support**: Process invoices in different languages
- **Layout Intelligence**: Handle various invoice designs and formats

### ICD-10 Code Selection Features
- **Interactive Interface**: User-friendly checkbox selection across 10 medical categories
- **Comprehensive Code Library**: 80+ common ICD-10 codes covering major medical conditions
- **AI-Powered Analysis**: Smart determination of applicable codes from medical notes
- **Evidence-Based Selection**: Direct quotes from medical notes supporting each code
- **Clinical Reasoning**: Detailed explanations for why each code was selected or rejected
- **Confidence Assessment**: High/Medium/Low confidence scoring for reliability
- **Category Classification**: Primary/Secondary/Comorbidity designation for each code

### Handwriting Recognition Features
- **Medical-Specialized OCR**: Advanced text extraction optimized for medical handwriting
- **Terminology Understanding**: Accurate recognition of medical terms, drug names, and abbreviations
- **Quality Assessment**: Comprehensive evaluation of legibility, ink clarity, and paper condition
- **Structure Analysis**: Automatic identification of note sections and formatting preservation
- **Correction Detection**: Recognition and tracking of crossed-out or modified text
- **Multiple Output Formats**: Structured JSON analysis or clean text extraction options
- **Digital Integration**: Export capabilities for EHR systems and electronic workflows

## Technical Specifications

- **Model**: Gemini 2.5 Flash (Preview)
- **Context Window**: 1,048,576 tokens
- **Audio Support**: Native audio processing with multiple formats
- **Image Support**: PNG, JPEG, WEBP, HEIC, HEIF formats
- **Processing**: Real-time streaming with progress indicators
- **Output**: Structured medical documentation and JSON data

## Security & Privacy

⚠️ **Important**: This is a demonstration application. For production use:

- Ensure HIPAA compliance
- Use secure, encrypted connections
- Implement proper access controls
- Follow medical data regulations
- Consider data residency requirements
- Implement proper financial data security

## Supported Formats

### Audio Formats
| Format | MIME Type | Description |
|--------|-----------|-------------|
| WAV | audio/wav | Uncompressed, high quality |
| MP3 | audio/mpeg | Compressed, widely supported |
| M4A | audio/m4a | Apple's compressed format |
| FLAC | audio/flac | Lossless compression |
| OGG | audio/ogg | Open source format |
| WEBM | audio/webm | Web-optimized format |

### Image Formats
| Format | MIME Type | Description |
|--------|-----------|-------------|
| PNG | image/png | Lossless compression, good for text |
| JPEG | image/jpeg | Standard photo format |
| WEBP | image/webp | Modern web format |
| HEIC | image/heic | Apple's high efficiency format |
| HEIF | image/heif | High efficiency image format |

## Use Cases

### Healthcare Applications
- **Patient Consultations**: Transcribe doctor-patient conversations
- **Medical Dictations**: Convert voice notes to structured text
- **Clinical Meetings**: Document team discussions and decisions
- **Procedure Notes**: Capture surgical or treatment procedures
- **Treatment Planning**: Record treatment discussions and plans

### Medical Coding Applications
- **Coding Assistance**: Support medical coders with automated suggestions
- **Quality Assurance**: Review clinical documentation for missed codes
- **Training and Education**: Help students learn medical coding
- **Billing Support**: Ensure accurate coding for insurance claims
- **Clinical Documentation Improvement**: Enhance documentation quality

### ICD-10 Code Selection Applications
- **Medical Coding Training**: Interactive learning tool for coding students and new coders
- **Code Validation**: Double-check manual code selections against AI recommendations
- **Documentation Review**: Identify potential codes that may have been missed
- **Coding Efficiency**: Speed up the coding process with pre-filtered relevant codes
- **Quality Assurance**: Ensure comprehensive code coverage for complex cases
- **Educational Support**: Learn clinical reasoning behind code selections

### Handwriting Recognition Applications
- **Digital Transformation**: Convert paper-based medical records to digital format
- **EHR Integration**: Prepare handwritten notes for electronic health record systems
- **Backup Documentation**: Create searchable digital copies of handwritten notes
- **Quality Improvement**: Standardize and improve readability of medical documentation
- **Accessibility Enhancement**: Make handwritten content accessible to screen readers
- **Archival Systems**: Digitize historical medical records for long-term storage
- **Mobile Documentation**: Process photos of handwritten notes taken with mobile devices

### Financial and Administrative Applications
- **Healthcare Billing**: Process medical service invoices
- **Expense Management**: Track healthcare-related expenses
- **Vendor Management**: Organize supplier and vendor information
- **Accounts Payable**: Streamline invoice processing workflows
- **Audit Documentation**: Maintain proper financial records

### Benefits
- **Time Saving**: Reduce manual transcription, coding, and data entry time
- **Accuracy**: Minimize human transcription, coding, and processing errors
- **Consistency**: Standardize medical documentation and financial data formats
- **Accessibility**: Make audio content searchable and accessible
- **Compliance**: Maintain proper medical documentation and financial standards

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure `GEMINI_API_KEY` is set correctly
   - Verify your API key is valid and active

2. **Audio Upload Issues**
   - Check file size (max 25MB)
   - Verify file format is supported
   - Ensure audio file is not corrupted

3. **Image Upload Issues**
   - Check file size (max 20MB)
   - Verify image format is supported
   - Ensure image is clear and readable
   - Try different lighting conditions

4. **Processing Errors**
   - Check internet connection
   - Verify API quota and limits
   - Try smaller files for testing

5. **JSON Parsing Errors**
   - Check if the response format is valid JSON
   - Review the raw response for formatting issues
   - Try simpler input content for testing

### Getting Help

If you encounter issues:
1. Check the Streamlit logs in the terminal
2. Verify your API key and internet connection
3. Try with smaller test files first
4. Consult the [Gemini documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash)
5. Review the [image understanding guide](https://ai.google.dev/gemini-api/docs/image-understanding)

## Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.5 Flash
- **Audio Processing**: Native Gemini audio capabilities
- **Image Processing**: Gemini image understanding + Pillow (PIL)
- **Data Processing**: Pandas for structured data handling
- **Language**: Python 3.8+

## Future Enhancements

Potential additions for future versions:
- Additional medical AI demos (radiology analysis, drug interaction checking)
- Real-time audio streaming
- Integration with EHR systems
- Advanced medical report templates
- Multi-language support
- Batch processing capabilities
- CPT code extraction
- Medical entity recognition
- Receipt and expense categorization
- Integration with accounting systems
- Handwriting style analysis and authentication
- Batch image processing for multiple notes
- Real-time handwriting recognition via camera

## License

This project is for educational and demonstration purposes. Please ensure compliance with all applicable healthcare regulations and privacy laws when using in production environments.

## Acknowledgments

- Google Gemini 2.5 Flash for advanced AI capabilities
- Streamlit for the web framework
- The healthcare AI research community for inspiring this demo

---

**Disclaimer**: This application is for demonstration purposes only. Always consult with qualified healthcare professionals for medical decisions and ensure compliance with healthcare regulations when processing real medical data. ICD codes should be verified by certified medical coding professionals before use in official documentation. Financial data should be reviewed for accuracy before use in accounting or billing systems. 