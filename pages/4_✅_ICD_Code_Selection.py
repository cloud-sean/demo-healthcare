import streamlit as st
import os
import json
from google import genai
from google.genai import types
import time
import pandas as pd

st.set_page_config(page_title="ICD Code Selection", page_icon="✅", layout="wide")

st.markdown("# ✅ ICD-10 Code Selection")
st.sidebar.header("ICD Code Selection Demo")

st.write(
    """
    This demo uses Gemini 2.5 Flash to automatically analyze medical notes and select relevant ICD-10 codes. 
    Enter your medical note below, and the AI will analyze it against our database of common ICD-10 codes, 
    automatically selecting the most appropriate ones with detailed reasoning and confidence scores.
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

# Common ICD-10 codes organized by category
@st.cache_data
def get_icd10_codes():
    """Return a structured list of common ICD-10 codes."""
    return {
        "Cardiovascular Diseases (I00-I99)": {
            "I10": "Essential (primary) hypertension",
            "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
            "I48.91": "Unspecified atrial fibrillation", 
            "I50.9": "Heart failure, unspecified",
            "I11.9": "Hypertensive heart disease without heart failure",
            "I25.9": "Chronic ischemic heart disease, unspecified",
            "I48.0": "Paroxysmal atrial fibrillation",
            "I20.9": "Angina pectoris, unspecified"
        },
        "Endocrine, Nutritional and Metabolic Diseases (E00-E89)": {
            "E11.9": "Type 2 diabetes mellitus without complications",
            "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
            "E11.40": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
            "E11.22": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
            "E10.9": "Type 1 diabetes mellitus without complications",
            "E78.5": "Hyperlipidemia, unspecified",
            "E66.9": "Obesity, unspecified",
            "E03.9": "Hypothyroidism, unspecified",
            "E05.90": "Thyrotoxicosis, unspecified without thyrotoxic crisis",
            "E87.6": "Hypokalemia"
        },
        "Mental, Behavioral and Neurodevelopmental Disorders (F01-F99)": {
            "F32.9": "Major depressive disorder, single episode, unspecified",
            "F33.9": "Major depressive disorder, recurrent, unspecified",
            "F41.9": "Anxiety disorder, unspecified",
            "F41.1": "Generalized anxiety disorder",
            "F10.20": "Alcohol dependence, uncomplicated",
            "F43.10": "Post-traumatic stress disorder, unspecified",
            "F84.0": "Autistic disorder",
            "F90.9": "Attention-deficit hyperactivity disorder, unspecified type",
            "F31.9": "Bipolar disorder, unspecified",
            "F20.9": "Schizophrenia, unspecified"
        },
        "Diseases of the Respiratory System (J00-J99)": {
            "J44.1": "Chronic obstructive pulmonary disease with acute exacerbation",
            "J45.9": "Asthma, unspecified",
            "J06.9": "Acute upper respiratory infection, unspecified",
            "J18.9": "Pneumonia, unspecified organism",
            "J44.0": "Chronic obstructive pulmonary disease with acute lower respiratory infection",
            "J20.9": "Acute bronchitis, unspecified",
            "J42": "Unspecified chronic bronchitis",
            "J43.9": "Emphysema, unspecified"
        },
        "Diseases of the Digestive System (K00-K95)": {
            "K21.9": "Gastro-esophageal reflux disease without esophagitis",
            "K59.00": "Constipation, unspecified",
            "K29.70": "Gastritis, unspecified, without bleeding",
            "K80.20": "Calculus of gallbladder without cholangitis or cholecystitis, without obstruction",
            "K57.90": "Diverticulosis of intestine, part unspecified, without perforation or abscess without bleeding",
            "K25.9": "Peptic ulcer, site unspecified, unspecified as acute or chronic, without hemorrhage or perforation",
            "K92.9": "Disease of digestive system, unspecified"
        },
        "Diseases of the Genitourinary System (N00-N99)": {
            "N18.6": "End stage renal disease",
            "N39.0": "Urinary tract infection, site not specified",
            "N40.1": "Enlarged prostate with lower urinary tract symptoms",
            "N18.3": "Chronic kidney disease, stage 3 (moderate)",
            "N20.0": "Calculus of kidney",
            "N93.9": "Abnormal uterine and vaginal bleeding, unspecified"
        },
        "Diseases of the Musculoskeletal System (M00-M99)": {
            "M25.512": "Pain in left shoulder",
            "M54.5": "Low back pain",
            "M79.3": "Panniculitis, unspecified",
            "M06.9": "Rheumatoid arthritis, unspecified",
            "M15.9": "Polyosteoarthritis, unspecified",
            "M54.2": "Cervicalgia"
        },
        "Diseases of the Nervous System (G00-G99)": {
            "G89.29": "Other chronic pain",
            "G43.909": "Migraine, unspecified, not intractable, without status migrainosus",
            "G40.909": "Epilepsy, unspecified, not intractable, without status epilepticus",
            "G20": "Parkinson's disease",
            "G30.9": "Alzheimer's disease, unspecified"
        },
        "Infectious and Parasitic Diseases (A00-B99)": {
            "B34.9": "Viral infection, unspecified",
            "A09": "Infectious gastroenteritis and colitis, unspecified",
            "B37.9": "Candidiasis, unspecified",
            "U07.1": "COVID-19",
            "A41.9": "Sepsis, unspecified organism"
        },
        "Symptoms, Signs and Abnormal Findings (R00-R99)": {
            "R50.9": "Fever, unspecified",
            "R06.02": "Shortness of breath",
            "R51": "Headache",
            "R42": "Dizziness and giddiness",
            "R05": "Cough",
            "R10.9": "Unspecified abdominal pain",
            "R53.83": "Fatigue"
        }
    }

def create_icd_analysis_prompt(all_codes, medical_note):
    """Create the prompt for ICD code analysis and selection."""
    # Flatten all codes into a single list
    codes_list = []
    for category, codes in all_codes.items():
        for code, description in codes.items():
            codes_list.append(f"- {code}: {description}")
    
    codes_string = "\n".join(codes_list)
    
    return f"""You are an expert medical coder specializing in ICD-10 code assignment. Analyze the provided medical note and determine which ICD-10 codes from the available list should be assigned to this patient.

AVAILABLE ICD-10 CODES:
{codes_string}

MEDICAL NOTE:
{medical_note}

Your task is to:
1. Carefully review the medical note for documented conditions, symptoms, and diagnoses
2. Select only the ICD-10 codes that are clearly supported by the medical documentation
3. Provide supporting evidence from the medical note for each selected code
4. Assign confidence levels and categorize each code appropriately

CODING GUIDELINES:
- Only select codes with clear documentation support
- "High" confidence: Condition explicitly documented
- "Medium" confidence: Condition strongly implied by symptoms/findings
- "Low" confidence: Some indication but limited evidence
- Primary: Main reason for encounter/most significant condition
- Secondary: Additional documented conditions
- Comorbidity: Pre-existing conditions that affect treatment

Return your analysis in the following JSON format:

{{
  "analysis_summary": {{
    "total_selected_codes": 0,
    "high_confidence_codes": 0,
    "medium_confidence_codes": 0,
    "low_confidence_codes": 0,
    "primary_diagnosis_suggested": "Most likely primary diagnosis based on the note"
  }},
  "selected_codes": [
    {{
      "code": "ICD-10 code",
      "description": "Full description",
      "confidence": "High/Medium/Low",
      "evidence": "Direct quotes from the medical note supporting this code",
      "clinical_reasoning": "Explanation of why this code applies and how it relates to the documentation",
      "category": "Primary/Secondary/Comorbidity"
    }}
  ],
  "codes_to_select": [
    "I10",
    "E11.9"
  ],
  "coding_notes": "Professional notes about the coding decisions, documentation quality, potential alternatives, or recommendations for additional documentation needed"
}}

IMPORTANT: The "codes_to_select" array should contain ONLY the ICD-10 code strings (like "I10", "E11.65") that should be automatically selected in the interface. This will be used to check the appropriate checkboxes automatically.

Be thorough, accurate, and conservative in your selections. Only include codes that are clearly justified by the medical documentation."""

def process_icd_analysis(client, all_codes, medical_note):
    """Process ICD code analysis using Gemini."""
    try:
        model = "gemini-2.5-flash-preview-05-20"
        
        # Create the prompt
        prompt = create_icd_analysis_prompt(all_codes, medical_note)
        
        # Prepare content
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
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
                progress = min(chunks_received * 0.1, 0.9)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing medical note for ICD codes... {len(response_text)} characters processed")
        
        progress_bar.progress(1.0)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Parse JSON response
        try:
            json_response = json.loads(response_text)
            return json_response
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {str(e)}")
            st.text("Raw response:")
            st.text(response_text[:1000] + "..." if len(response_text) > 1000 else response_text)
            return None
                
    except Exception as e:
        st.error(f"Error processing ICD analysis: {str(e)}")
        return None

def display_analysis_results(results):
    """Display the ICD code analysis results."""
    if not results:
        return
    
    # Analysis Summary
    summary = results.get("analysis_summary", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Selected Codes", summary.get("total_selected_codes", 0))
    with col2:
        st.metric("High Confidence", summary.get("high_confidence_codes", 0))
    with col3:
        st.metric("Medium Confidence", summary.get("medium_confidence_codes", 0))
    with col4:
        st.metric("Low Confidence", summary.get("low_confidence_codes", 0))
    
    # Primary Diagnosis
    if summary.get("primary_diagnosis_suggested"):
        st.info(f"**Suggested Primary Diagnosis:** {summary['primary_diagnosis_suggested']}")
    
    # Selected Codes
    selected_codes = results.get("selected_codes", [])
    if selected_codes:
        st.subheader("🎯 Selected ICD-10 Codes")
        
        for code_info in selected_codes:
            confidence = code_info.get("confidence", "Unknown")
            category = code_info.get("category", "Unknown")
            
            # Color code by confidence
            if confidence == "High":
                st.success(f"**{code_info.get('code', 'N/A')}** - {code_info.get('description', 'N/A')}")
            elif confidence == "Medium":
                st.warning(f"**{code_info.get('code', 'N/A')}** - {code_info.get('description', 'N/A')}")
            else:
                st.error(f"**{code_info.get('code', 'N/A')}** - {code_info.get('description', 'N/A')}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(f"**Confidence:** {confidence}")
                st.write(f"**Category:** {category}")
            
            with col2:
                st.write(f"**Evidence:** {code_info.get('evidence', 'N/A')}")
            
            st.write(f"**Clinical Reasoning:** {code_info.get('clinical_reasoning', 'N/A')}")
            st.markdown("---")
    
    # Coding Notes
    if results.get("coding_notes"):
        st.subheader("📝 Coding Notes")
        st.info(results["coding_notes"])
    
    # Download Options
    st.subheader("📄 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON download
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="📄 Download Full Analysis (JSON)",
            data=json_str,
            file_name=f"icd_analysis_{int(time.time())}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV of selected codes
        if selected_codes:
            df_data = []
            for code in selected_codes:
                df_data.append({
                    "ICD-10 Code": code.get("code", ""),
                    "Description": code.get("description", ""),
                    "Confidence": code.get("confidence", ""),
                    "Category": code.get("category", ""),
                    "Evidence": code.get("evidence", ""),
                    "Clinical Reasoning": code.get("clinical_reasoning", "")
                })
            
            df = pd.DataFrame(df_data)
            csv_str = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Selected Codes (CSV)",
                data=csv_str,
                file_name=f"selected_icd_codes_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Summary report
        summary_report = f"""ICD-10 Code Analysis Summary
================================

Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Selected Codes: {len(selected_codes)}
High Confidence: {summary.get('high_confidence_codes', 0)}
Medium Confidence: {summary.get('medium_confidence_codes', 0)}
Low Confidence: {summary.get('low_confidence_codes', 0)}

Primary Diagnosis: {summary.get('primary_diagnosis_suggested', 'Not specified')}

Selected Codes:
{chr(10).join([f"- {code.get('code', 'N/A')}: {code.get('description', 'N/A')} ({code.get('confidence', 'Unknown')} confidence)" for code in selected_codes])}

Coding Notes:
{results.get('coding_notes', 'None')}
"""
        
        st.download_button(
            label="📋 Download Summary Report",
            data=summary_report,
            file_name=f"icd_analysis_summary_{int(time.time())}.txt",
            mime="text/plain"
        )

# Initialize client
client = get_gemini_client()
if not client:
    st.stop()

# Get ICD codes
icd_codes = get_icd10_codes()

# Medical Note Input (Step 1)
st.subheader("Step 1: Enter Medical Note")
st.write("Provide the medical note or clinical documentation. The AI will analyze it and automatically select relevant ICD-10 codes.")

medical_note = st.text_area(
    "Medical Note",
    height=200,
    placeholder="""Example:
Chief Complaint: Chest pain and shortness of breath

History of Present Illness: 
65-year-old male presents with acute onset chest pain started 2 hours ago. Pain is described as crushing, substernal, radiating to left arm. Associated with shortness of breath, diaphoresis, and nausea. Patient has history of hypertension and type 2 diabetes mellitus.

Physical Examination:
BP: 180/100 mmHg, HR: 110 bpm, RR: 24, O2 Sat: 92% on room air
Heart: Tachycardic, no murmurs
Lungs: Bilateral rales in lower lobes

Assessment and Plan:
1. Acute coronary syndrome - EKG, cardiac enzymes, cardiology consult
2. Hypertensive crisis - IV antihypertensives
3. Type 2 diabetes - continue metformin, monitor glucose
4. Acute heart failure - diuretics, monitor I/O"""
)

# Process button
if st.button("🧠 Analyze Medical Note for ICD Codes", type="primary", disabled=not medical_note.strip()):
    if not medical_note.strip():
        st.error("Please enter a medical note.")
    else:
        with st.spinner("Analyzing medical note for ICD code selection... This may take a moment."):
            result = process_icd_analysis(client, icd_codes, medical_note)
            
            if result:
                st.session_state.analysis_result = result
                
                # Update checkbox states based on AI selection
                codes_to_select = result.get("codes_to_select", [])
                
                # Clear all previous selections
                for category_codes in icd_codes.values():
                    for code in category_codes.keys():
                        st.session_state[f"code_{code}"] = False
                
                # Set selected codes
                for code in codes_to_select:
                    st.session_state[f"code_{code}"] = True
                
                st.success(f"✅ Analysis completed! {len(codes_to_select)} ICD-10 codes automatically selected.")

# Step 2: Show Selected Codes
st.subheader("Step 2: AI-Selected ICD-10 Codes")
st.write("The checkboxes below show the codes automatically selected by the AI based on your medical note.")

# Create tabs for different code categories
categories = list(icd_codes.keys())
tabs = st.tabs(categories)

ai_selected_codes = {}

for i, category in enumerate(categories):
    with tabs[i]:
        st.write(f"**{category}**")
        
        # Show checkboxes (will be automatically checked based on AI selection)
        for code, description in icd_codes[category].items():
            is_selected = st.session_state.get(f"code_{code}", False)
            
            # Show checkbox (read-only display of AI selection)
            checkbox_value = st.checkbox(
                f"{code} - {description}", 
                value=is_selected,
                key=f"display_code_{code}",
                disabled=True  # Make read-only to show AI selection
            )
            
            if is_selected:
                ai_selected_codes[code] = description

# Display analysis results
if hasattr(st.session_state, 'analysis_result') and st.session_state.analysis_result:
    st.markdown("---")
    st.subheader("🎯 ICD Code Analysis Results")
    
    if ai_selected_codes:
        st.success(f"✅ {len(ai_selected_codes)} codes selected automatically by AI")
        
        # Show selected codes summary
        with st.expander("📋 View Selected Codes Summary", expanded=True):
            for code, description in ai_selected_codes.items():
                st.write(f"• **{code}**: {description}")
    else:
        st.info("No ICD-10 codes were selected by the AI for this medical note.")
    
    display_analysis_results(st.session_state.analysis_result)
    
    # Clear results button
    if st.button("🗑️ Clear Analysis and Reset"):
        if 'analysis_result' in st.session_state:
            del st.session_state.analysis_result
        
        # Clear all checkbox states
        for category_codes in icd_codes.values():
            for code in category_codes.keys():
                if f"code_{code}" in st.session_state:
                    del st.session_state[f"code_{code}"]
        
        st.rerun()

# Sidebar information
with st.sidebar:
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **📝 Enter Medical Note**: Provide clinical documentation
    2. **🧠 AI Analysis**: Gemini analyzes against 80+ ICD codes  
    3. **✅ Automatic Selection**: Relevant codes are selected automatically
    4. **📊 Review Results**: See evidence and confidence scores
    5. **📄 Export**: Download analysis in multiple formats
    """)
    
    st.markdown("### 🔧 Features")
    st.markdown("""
    - **🤖 Automatic Selection**: AI chooses relevant codes
    - **🎯 Evidence-Based**: Supporting quotes for each code  
    - **📊 Confidence Scoring**: High/Medium/Low levels
    - **🏷️ Categorization**: Primary/Secondary/Comorbidity
    - **📝 Clinical Reasoning**: Detailed explanations
    - **📄 Export Options**: JSON, CSV, and summary downloads
    """)
    
    st.markdown("### 📋 Code Coverage")
    st.markdown("""
    **80+ codes across 10 categories**:
    - Cardiovascular (8 codes)
    - Endocrine/Metabolic (10 codes)  
    - Mental Health (10 codes)
    - Respiratory (8 codes)
    - Digestive System (7 codes)
    - Genitourinary (6 codes)
    - Musculoskeletal (6 codes)
    - Nervous System (5 codes)
    - Infectious Diseases (5 codes)
    - Symptoms & Signs (7 codes)
    """)
    
    st.markdown("### 🏥 Use Cases")
    st.markdown("""
    - **Coding Assistance**: AI-powered code suggestions
    - **Quality Assurance**: Verify manual coding
    - **Education**: Learn ICD coding patterns
    - **Documentation Review**: Identify missing codes
    - **Efficiency**: Speed up coding workflow
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Provide complete medical notes
    - Include chief complaint and assessment
    - Document all relevant conditions clearly
    - Use standard medical terminology
    - Include physical exam findings
    """)
    
    st.markdown("### ⚠️ Important Notice")
    st.warning("""
    **AI-Assisted Coding Tool**. 
    
    All ICD-10 code selections should be:
    - Verified by certified medical coders
    - Reviewed for accuracy and completeness  
    - Validated against current guidelines
    - Approved by qualified professionals
    """)
    
    st.markdown("### 🔗 Resources")
    st.markdown("""
    - [ICD-10-CM Guidelines](https://www.cms.gov/medicare/coordination-benefits-recovery/overview/icd-code-lists)
    - [WHO ICD-10 Reference](https://platform.who.int/mortality/about/list-of-causes-and-corresponding-icd-10-codes)
    - [CMS Coding Resources](https://www.cms.gov/medicare/coding-billing/icd-10-codes)
    """) 