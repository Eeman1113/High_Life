import streamlit as st
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
import io
import requests
import json
import time
import re
import fitz  # PyMuPDF
import tempfile
import os

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n\n"
    return text

def chunk_text(text, max_chunk_size=3000):
    """Split text into manageable chunks"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def get_key_phrases_for_chunk(chunk, api_key, retry_count=3):
    """Send a chunk to LLM and get key phrases to highlight"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an assistant that identifies the most important phrases and sentences in text."},
            {"role": "user", "content": f"""
                Analyze the following text section and extract ONLY the most important phrases, 
                sentences, or key points that should be highlighted. Return EXACTLY the phrases 
                as they appear in the original text, one per line, with no bullets, numbers, or 
                other formatting. ONLY include text that appears verbatim in the original document.
                
                TEXT SECTION:
                {chunk}
            """}
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
        "top_p": 1
    }
    
    for attempt in range(retry_count):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip().split('\n')
        elif response.status_code == 429:
            # Rate limit hit - extract wait time if possible
            wait_time = 10  # Default wait time
            try:
                error_message = response.json().get("error", {}).get("message", "")
                wait_match = re.search(r"try again in (\d+\.\d+)s", error_message)
                if wait_match:
                    wait_time = float(wait_match.group(1)) + 1  # Add a buffer
            except Exception:
                pass
            
            st.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds before retrying...")
            time.sleep(wait_time)
            
            if attempt == retry_count - 1:
                return []
        else:
            st.error(f"API Error: {response.status_code}")
            st.error(response.text)
            return []
    
    return []

def highlight_pdf(input_pdf_bytes, key_phrases):
    """Highlight key phrases in the PDF with yellow highlighting"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(input_pdf_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(tmp_path)
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Search for each key phrase and highlight it
            for phrase in key_phrases:
                phrase = phrase.strip()
                if len(phrase) > 5:  # Only highlight meaningful phrases
                    instances = page.search_for(phrase)
                    for inst in instances:
                        # Add yellow highlight annotation
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors({"stroke": (1, 1, 0)})  # Yellow
                        highlight.update()
        
        # Save to a new BytesIO object
        output_bytes = io.BytesIO()
        doc.save(output_bytes)
        doc.close()
        
        # Return as bytes
        output_bytes.seek(0)
        return output_bytes.getvalue()
    
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

def main():
    st.title("PDF Highlighter")
    st.write("Upload a PDF to highlight important information")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        try:
            # Save original PDF bytes
            pdf_bytes = uploaded_file.getvalue()
            
            # Process the file
            with st.spinner("Extracting text from PDF..."):
                # Extract text
                text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
                text_length = len(text)
                
                # Show text stats
                st.info(f"Extracted {text_length} characters from PDF")
                
                # Show extracted text in an expandable section
                with st.expander("View extracted text"):
                    st.text(text)
            
            # API key directly in the code
            api_key = "YOUR KEY HERE"
            
            # Chunk the text
            with st.spinner("Splitting document into processable chunks..."):
                chunks = chunk_text(text, max_chunk_size=3000)
                st.info(f"Document split into {len(chunks)} chunks for processing")
                
                # Warning for large documents
                if len(chunks) > 5:
                    st.warning(f"This is a large document with {len(chunks)} sections. Processing may take some time due to API rate limits.")
            
            # Only proceed if user confirms for large documents
            proceed = True
            if len(chunks) > 10:
                proceed = st.button("This is a large document. Click to start processing")
            
            if proceed:
                # Create a progress bar
                st.write("Processing document chunks...")
                progress_bar = st.progress(0)
                
                # Process each chunk to get key phrases
                all_key_phrases = []
                
                for i, chunk in enumerate(chunks):
                    # Update progress
                    progress_bar.progress((i + 1) / len(chunks))
                    
                    # Show current chunk being processed
                    st.info(f"Processing chunk {i+1} of {len(chunks)}...")
                    
                    # Get key phrases to highlight
                    key_phrases = get_key_phrases_for_chunk(chunk, api_key)
                    all_key_phrases.extend(key_phrases)
                    
                    # Add a small delay between chunks to help with rate limiting
                    if i < len(chunks) - 1:
                        time.sleep(1)
                
                # Highlight the PDF
                with st.spinner("Creating highlighted PDF..."):
                    highlighted_pdf = highlight_pdf(pdf_bytes, all_key_phrases)
                
                # Display download button for highlighted PDF
                st.success("PDF processing complete! Key information has been highlighted in yellow.")
                st.download_button(
                    label="Download Highlighted PDF",
                    data=highlighted_pdf,
                    file_name="highlighted_document.pdf",
                    mime="application/pdf"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
