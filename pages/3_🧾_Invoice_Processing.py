import streamlit as st
import base64
import os
import json
from google import genai
from google.genai import types
import time
import pandas as pd
from PIL import Image
import io

st.set_page_config(page_title="Invoice Processing", page_icon="🧾", layout="wide")

st.markdown("# 🧾 Invoice Processing")
st.sidebar.header("Invoice Processing Demo")

st.write(
    """
    This demo uses Gemini 2.5 Flash's advanced image understanding to automatically extract structured data from invoice images and return JSON responses. Upload an invoice and get back organized JSON data 
    perfect for accounting systems, expense tracking, and automated bookkeeping.
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

def invoice_extraction_prompt():
    """Return the invoice processing prompt with JSON schema."""
    return """You are an expert in invoice processing and data extraction. Analyze the provided invoice image and extract all relevant information into a structured JSON format.

Please extract the following information:

1. **Vendor Information**: Company name, address, contact details, tax ID
2. **Invoice Details**: Invoice number, date, due date, purchase order number
3. **Billing Information**: Bill-to company/person, address
4. **Line Items**: All products/services with descriptions, quantities, unit prices, totals
5. **Financial Summary**: Subtotal, taxes, discounts, total amount
6. **Payment Information**: Payment terms, methods accepted, bank details if present

Return the response in the following JSON format:

{
  "invoice_metadata": {
    "extraction_confidence": "High/Medium/Low",
    "invoice_type": "Standard Invoice/Receipt/Credit Note/Other",
    "currency": "USD/EUR/GBP/etc",
    "language": "English/Spanish/French/etc"
  },
  "vendor": {
    "name": "Company name",
    "address": {
      "street": "Street address",
      "city": "City",
      "state": "State/Province",
      "postal_code": "ZIP/Postal code",
      "country": "Country"
    },
    "contact": {
      "phone": "Phone number",
      "email": "Email address",
      "website": "Website URL"
    },
    "tax_id": "Tax ID/VAT number"
  },
  "invoice_details": {
    "invoice_number": "Invoice number",
    "invoice_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD",
    "purchase_order": "PO number if present"
  },
  "billing_to": {
    "name": "Bill-to name",
    "address": {
      "street": "Street address",
      "city": "City",
      "state": "State/Province", 
      "postal_code": "ZIP/Postal code",
      "country": "Country"
    }
  },
  "line_items": [
    {
      "description": "Product/service description",
      "quantity": 0,
      "unit_price": 0.00,
      "total": 0.00,
      "tax_rate": "Tax percentage if specified",
      "category": "Product/Service category if identifiable"
    }
  ],
  "financial_summary": {
    "subtotal": 0.00,
    "tax_amount": 0.00,
    "discount_amount": 0.00,
    "shipping_amount": 0.00,
    "total_amount": 0.00
  },
  "payment_info": {
    "payment_terms": "Net 30/Due on receipt/etc",
    "payment_methods": ["Cash", "Check", "Credit Card", "Bank Transfer"],
    "bank_details": {
      "account_name": "Account name if present",
      "account_number": "Account number if present",
      "routing_number": "Routing number if present",
      "swift_code": "SWIFT code if present"
    }
  },
  "additional_notes": "Any special instructions, terms, or notes on the invoice",
  "extracted_text_confidence": "Assessment of text recognition quality"
}

Be thorough and accurate. If information is not clearly visible or present, use null or empty values. Ensure all numerical values are properly formatted as numbers, not strings."""

def get_image_mime_type(filename):
    """Get MIME type for image file."""
    extension = filename.lower().split('.')[-1]
    mime_types = {
        'png': 'image/png',
        'jpg': 'image/jpeg', 
        'jpeg': 'image/jpeg',
        'webp': 'image/webp',
        'heic': 'image/heic',
        'heif': 'image/heif'
    }
    return mime_types.get(extension, 'image/jpeg')

def process_invoice_image(client, image_file):
    """Process invoice image using Gemini with JSON output."""
    try:
        model = "gemini-2.5-flash-preview-05-20"
        
        # Read image file and create proper image part
        image_bytes = image_file.read()
        mime_type = get_image_mime_type(image_file.name)
        
        # Prepare content
        parts = [
            types.Part.from_text(text=invoice_extraction_prompt()),
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
        ]
        
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
                progress = min(chunks_received * 0.1, 0.9)
                progress_bar.progress(progress)
                status_text.text(f"Processing invoice image... {len(response_text)} characters extracted")
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
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
            st.text(response_text[:1000] + "..." if len(response_text) > 1000 else response_text)
            return None
                
    except Exception as e:
        st.error(f"Error processing invoice: {str(e)}")
        return None

def display_invoice_results(results):
    """Display invoice processing results in an organized format."""
    if not results:
        return
    
    # Display metadata and confidence
    metadata = results.get("invoice_metadata", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence = metadata.get("extraction_confidence", "Unknown")
        if confidence == "High":
            st.success(f"🟢 Confidence: {confidence}")
        elif confidence == "Medium":
            st.warning(f"🟡 Confidence: {confidence}")
        else:
            st.error(f"🔴 Confidence: {confidence}")
    
    with col2:
        invoice_type = metadata.get("invoice_type", "Unknown")
        st.info(f"📄 Type: {invoice_type}")
    
    with col3:
        currency = metadata.get("currency", "Unknown")
        st.info(f"💰 Currency: {currency}")
    
    with col4:
        language = metadata.get("language", "Unknown")
        st.info(f"🗣️ Language: {language}")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🏢 Vendor Info", "📋 Line Items", "💳 Payment Info"])
    
    with tab1:
        st.subheader("Invoice Summary")
        
        # Invoice details
        invoice_details = results.get("invoice_details", {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Invoice Details:**")
            st.write(f"• **Number:** {invoice_details.get('invoice_number', 'N/A')}")
            st.write(f"• **Date:** {invoice_details.get('invoice_date', 'N/A')}")
            st.write(f"• **Due Date:** {invoice_details.get('due_date', 'N/A')}")
            st.write(f"• **PO Number:** {invoice_details.get('purchase_order', 'N/A')}")
        
        with col2:
            # Financial summary
            financial = results.get("financial_summary", {})
            st.markdown("**Financial Summary:**")
            st.write(f"• **Subtotal:** ${financial.get('subtotal', 0):.2f}")
            st.write(f"• **Tax:** ${financial.get('tax_amount', 0):.2f}")
            st.write(f"• **Discount:** ${financial.get('discount_amount', 0):.2f}")
            st.write(f"• **Shipping:** ${financial.get('shipping_amount', 0):.2f}")
            st.markdown(f"**• Total: ${financial.get('total_amount', 0):.2f}**")
        
        # Billing information
        billing = results.get("billing_to", {})
        if billing and billing.get("name"):
            st.markdown("**Bill To:**")
            st.write(f"**{billing.get('name', 'N/A')}**")
            address = billing.get("address", {})
            if address:
                address_parts = [
                    address.get("street", ""),
                    address.get("city", ""),
                    address.get("state", ""),
                    address.get("postal_code", ""),
                    address.get("country", "")
                ]
                full_address = ", ".join([part for part in address_parts if part])
                if full_address:
                    st.write(full_address)
    
    with tab2:
        st.subheader("Vendor Information")
        
        vendor = results.get("vendor", {})
        
        # Company name
        if vendor.get("name"):
            st.markdown(f"### {vendor['name']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Address
            address = vendor.get("address", {})
            if address:
                st.markdown("**Address:**")
                address_parts = [
                    address.get("street", ""),
                    address.get("city", ""),
                    address.get("state", ""),
                    address.get("postal_code", ""),
                    address.get("country", "")
                ]
                for part in address_parts:
                    if part:
                        st.write(part)
        
        with col2:
            # Contact information
            contact = vendor.get("contact", {})
            if contact:
                st.markdown("**Contact:**")
                if contact.get("phone"):
                    st.write(f"📞 {contact['phone']}")
                if contact.get("email"):
                    st.write(f"📧 {contact['email']}")
                if contact.get("website"):
                    st.write(f"🌐 {contact['website']}")
            
            # Tax ID
            if vendor.get("tax_id"):
                st.markdown("**Tax ID:**")
                st.write(vendor["tax_id"])
    
    with tab3:
        st.subheader("Line Items")
        
        line_items = results.get("line_items", [])
        if line_items:
            # Create DataFrame for better display
            df_data = []
            for item in line_items:
                df_data.append({
                    "Description": item.get("description", ""),
                    "Quantity": item.get("quantity", 0),
                    "Unit Price": f"${item.get('unit_price', 0):.2f}",
                    "Total": f"${item.get('total', 0):.2f}",
                    "Category": item.get("category", "N/A")
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Summary stats
            total_items = len(line_items)
            total_quantity = sum(item.get("quantity", 0) for item in line_items)
            st.write(f"**Total Items:** {total_items} | **Total Quantity:** {total_quantity}")
        else:
            st.info("No line items found in the invoice.")
    
    with tab4:
        st.subheader("Payment Information")
        
        payment_info = results.get("payment_info", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            if payment_info.get("payment_terms"):
                st.markdown("**Payment Terms:**")
                st.write(payment_info["payment_terms"])
            
            payment_methods = payment_info.get("payment_methods", [])
            if payment_methods:
                st.markdown("**Accepted Payment Methods:**")
                for method in payment_methods:
                    st.write(f"• {method}")
        
        with col2:
            bank_details = payment_info.get("bank_details", {})
            if any(bank_details.values()):
                st.markdown("**Bank Details:**")
                if bank_details.get("account_name"):
                    st.write(f"• **Account Name:** {bank_details['account_name']}")
                if bank_details.get("account_number"):
                    st.write(f"• **Account Number:** {bank_details['account_number']}")
                if bank_details.get("routing_number"):
                    st.write(f"• **Routing Number:** {bank_details['routing_number']}")
                if bank_details.get("swift_code"):
                    st.write(f"• **SWIFT Code:** {bank_details['swift_code']}")
    
    # Additional notes
    if results.get("additional_notes"):
        st.subheader("📝 Additional Notes")
        st.info(results["additional_notes"])
    
    # Download options
    st.subheader("📄 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON download
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name=f"invoice_data_{int(time.time())}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV download (line items)
        line_items = results.get("line_items", [])
        if line_items:
            df_data = []
            for item in line_items:
                df_data.append({
                    "Description": item.get("description", ""),
                    "Quantity": item.get("quantity", 0),
                    "Unit Price": item.get("unit_price", 0),
                    "Total": item.get("total", 0),
                    "Category": item.get("category", "")
                })
            
            df = pd.DataFrame(df_data)
            csv_str = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Line Items CSV",
                data=csv_str,
                file_name=f"invoice_line_items_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Summary report
        summary_report = f"""Invoice Processing Summary
================================

Invoice Number: {results.get('invoice_details', {}).get('invoice_number', 'N/A')}
Date: {results.get('invoice_details', {}).get('invoice_date', 'N/A')}
Vendor: {results.get('vendor', {}).get('name', 'N/A')}
Total Amount: ${results.get('financial_summary', {}).get('total_amount', 0):.2f}
Confidence: {results.get('invoice_metadata', {}).get('extraction_confidence', 'Unknown')}

Line Items: {len(results.get('line_items', []))}
Currency: {results.get('invoice_metadata', {}).get('currency', 'Unknown')}
"""
        
        st.download_button(
            label="📋 Download Summary",
            data=summary_report,
            file_name=f"invoice_summary_{int(time.time())}.txt",
            mime="text/plain"
        )

# Initialize client
client = get_gemini_client()
if not client:
    st.stop()

# Main interface
st.subheader("Upload Invoice Image")
st.write("Upload an invoice image to automatically extract structured data.")

# Image file uploader
image_file = st.file_uploader(
    "Choose an invoice image",
    type=['png', 'jpg', 'jpeg', 'webp', 'heic', 'heif'],
    help="Supported formats: PNG, JPG, JPEG, WEBP, HEIC, HEIF (Max: 20MB per file)"
)

if image_file is not None:
    # Display the image
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show the uploaded image
        try:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Invoice", use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
    
    with col2:
        # Show file info
        file_size = len(image_file.getvalue()) / (1024 * 1024)  # Size in MB
        st.info(f"📁 **File**: {image_file.name}")
        st.info(f"📏 **Size**: {file_size:.2f} MB")
        st.info(f"🖼️ **Format**: {image_file.type}")
        
        # Process button
        if st.button("🧾 Process Invoice", type="primary", use_container_width=True):
            with st.spinner("Processing invoice image... This may take a moment."):
                # Reset file pointer
                image_file.seek(0)
                
                result = process_invoice_image(client, image_file)
                
                if result:
                    st.session_state.invoice_result = result
                    st.success("✅ Invoice processing completed!")

# Display results
if hasattr(st.session_state, 'invoice_result') and st.session_state.invoice_result:
    st.markdown("---")
    st.subheader("🎯 Invoice Processing Results")
    
    display_invoice_results(st.session_state.invoice_result)
    
    # Clear results button
    if st.button("🗑️ Clear Results"):
        if 'invoice_result' in st.session_state:
            del st.session_state.invoice_result
        st.rerun()

# Sidebar information
with st.sidebar:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **🧾 Smart Extraction**: Automatic invoice data extraction
    - **💰 Financial Analysis**: Complete cost breakdown
    - **🏢 Vendor Details**: Company and contact information
    - **📋 Line Items**: Detailed product/service listings
    - **💳 Payment Info**: Terms and banking details
    - **📊 Export Options**: JSON, CSV, and summary reports
    """)
    
    st.markdown("### 📸 Supported Formats")
    st.markdown("""
    **Images**: PNG, JPG, JPEG, WEBP, HEIC, HEIF
    
    **Invoice Types**:
    - Standard invoices
    - Receipts
    - Credit notes
    - Purchase orders
    - Service bills
    """)
    
    st.markdown("### 🏭 Use Cases")
    st.markdown("""
    - Automated accounting
    - Expense tracking
    - Accounts payable processing
    - Tax preparation
    - Audit documentation
    - Vendor management
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Use clear, well-lit images
    - Ensure text is readable
    - Avoid shadows or glare
    - Capture the entire invoice
    - Use high resolution when possible
    """)
    
    st.markdown("### ⚠️ Privacy Notice")
    st.warning("""
    This is a demonstration tool. For production use:
    - Ensure data security compliance
    - Review extracted data for accuracy
    - Implement proper access controls
    - Follow accounting standards
    """)
    
    st.markdown("### 🔗 Learn More")
    st.markdown("""
    - [Gemini Image Understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
    - [Invoice Processing Best Practices](https://ai.google.dev/gemini-api/docs/image-understanding)
    - [Structured Data Extraction](https://ai.google.dev/gemini-api/docs/image-understanding)
    """) 