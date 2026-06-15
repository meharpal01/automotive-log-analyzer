# import os
# import re
# import io
# import torch
# import streamlit as st
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel

# # Set page configuration
# st.set_page_config(page_title="Automotive Log Analyzer", page_icon="🚗", layout="wide")

# # ==========================================
# # 1. Model Loading Cache Logic
# # ==========================================
# @st.cache_resource
# def load_fine_tuned_model():
#     """Loads and caches the model/tokenizer so it only runs once on startup."""
#     load_dotenv()
#     token = os.getenv("HF_TOKEN")
    
#     if not token:
#         st.error("⚠️ HF_TOKEN not found in your environment secrets or .env file.")
#         st.stop()
        
#     base = AutoModelForCausalLM.from_pretrained(
#         "google/gemma-2b", 
#         dtype=torch.float32, 
#         token=token
#     )
#     tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=token)
#     model = PeftModel.from_pretrained(base, ".")
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     return model, tokenizer, device

# # App title and description
# st.title("🚗 Automotive Validation Log Analyzer")
# st.markdown("Upload your validation logs file to generate structured diagnostic reports using your fine-tuned Gemma model.")

# # Initialize model loading with a visual spinner status tracker
# with st.spinner("📦 Loading base model and fine-tuned LoRA adapters into memory... Please wait..."):
#     model, tokenizer, device = load_fine_tuned_model()
# st.success("🤖 AI Engine Loaded successfully!")

# st.divider()

# # ==========================================
# # 2. Sidebar Configuration & File Upload
# # ==========================================
# st.sidebar.header("📁 Upload Options")
# uploaded_file = st.sidebar.file_uploader(
#     "Choose a validation log text file", 
#     type=["txt"], 
#     help="Upload the file containing logs formatted with 'INPUT LOG X' boundaries."
# )

# max_tokens = st.sidebar.slider("Max New Tokens", min_value=50, max_value=300, value=150, step=10)

# # ==========================================
# # 3. Processing and Inference Layout
# # ==========================================
# if uploaded_file is not None:
#     # Read the text string from the uploaded file buffer object safely
#     stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
#     file_content = stringio.read()
    
#     # Split text chunks precisely on the established boundary tags
#     log_blocks = re.split(r'={5,}\s*INPUT LOG \d+\s*={5,}', file_content)
    
#     # Clean up segments
#     valid_blocks = []
#     for block in log_blocks:
#         clean_block = block.strip()
#         clean_block = re.sub(r'^={5,}\s*|={5,}\s*$', '', clean_block).strip()
#         if clean_block:
#             valid_blocks.append(clean_block)
            
#     total_logs = len(valid_blocks)
#     st.info(f"📊 Identified **{total_logs}** individual log segments inside the file.")
    
#     # Trigger button to execute batch processing
#     if st.button("🚀 Execute Analysis Pipeline", type="primary"):
#         output_reports = []
        
#         # UI components for tracking execution progress interactively
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         # Visual display container blocks for live updates
#         results_container = st.container()
        
#         with results_container:
#             st.subheader("📋 Generated Analyses")
            
#             for idx, clean_block in enumerate(valid_blocks, start=1):
#                 # UI progress math step updates
#                 status_text.text(f"Processing Log {idx} of {total_logs}...")
#                 progress_bar.progress(idx / total_logs)
                
#                 # Dynamic context template preparation targeting the LoRA schema
#                 if "### Instruction:" not in clean_block:
#                     prompt = f"### Instruction:\n{clean_block}\n\n### Diagnostic Report Output:\n"
#                 else:
#                     if "### Diagnostic Report Output:" not in clean_block:
#                         prompt = f"{clean_block}\n\n### Diagnostic Report Output:\n"
#                     else:
#                         prompt = clean_block
                        
#                 # Perform AI Inference
#                 inputs = tokenizer(prompt, return_tensors="pt")
#                 inputs = {k: v.to(device) for k, v in inputs.items()} 
                
#                 with torch.no_grad():
#                     out = model.generate(**inputs, max_new_tokens=max_tokens)
                    
#                 response = tokenizer.decode(out[0], skip_special_tokens=True)
                
#                 # Append result to our master data tracker for downloads
#                 output_reports.append(response)
                
#                 # Render results to UI live as they finish
#                 with st.expander(f"✅ Log #{idx} Analysis Output Summary", expanded=True):
#                     st.code(response, language="markdown")
                    
#         # Update progress flags to completed state
#         progress_bar.empty()
#         status_text.text("🎉 Analysis complete for all items!")
        
#         # Join all resulting logs together with standard aesthetic clean split buffers
#         final_download_string = "\n\n" + "="*50 + "\n\n".join(output_reports)
        
#         # ==========================================
#         # 4. Download Result Generation Block
#         # ==========================================
#         st.divider()
#         st.subheader("📥 Export Diagnostic Reports")
        
#         st.download_button(
#             label="💾 Download All Reports as Text File",
#             data=final_download_string,
#             file_name="automotive_diagnostic_reports.txt",
#             mime="text/plain",
#             help="Click here to save the complete batch processing execution text string onto your local disk computer workspace."
#         )
# else:
#     # Landing page state warning placeholder block
#     st.warning("👈 Please upload a valid `.txt` log file in the sidebar menu panel to get started.")

import os
import re
import io
import torch
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Set clean page layout
st.set_page_config(page_title="AutoDiagGPT", page_icon="🚗", layout="centered")

# ==========================================
# 1. Model Loading (Runs Once on Startup)
# ==========================================
@st.cache_resource
def load_fine_tuned_model():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    
    if not token:
        st.error("⚠️ HF_TOKEN not found in your environment secrets or .env file.")
        st.stop()
        
    # base = AutoModelForCausalLM.from_pretrained(
    #     "google/gemma-2b", 
    #     dtype=torch.float32, 
    #     token=token
    # )
    # Optimized for Streamlit Cloud Free Tier (prevents RAM OOM crashes)
    base = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b", 
        dtype=torch.float16,           # Reduces memory usage by half compared to float32
        low_cpu_mem_usage=True,        # Streams layers sequentially instead of caching all at once
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=token)
    model = PeftModel.from_pretrained(base, ".")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device
# App title
st.title("🚗 AutoDiagGPT")
st.markdown("Upload your validation logs file to generate structured diagnostic reports instantly.")

# Visual spinner while loading components
with st.spinner("📦 Preparing AI Engine... Please wait..."):
    model, tokenizer, device = load_fine_tuned_model()

st.divider()

# ==========================================
# 2. Simplified File Upload
# ==========================================
uploaded_file = st.file_uploader(
    "Upload your validation log text file (.txt)", 
    type=["txt"]
)

# ==========================================
# 3. Processing and Inference Layout
# ==========================================
if uploaded_file is not None:
    # Read the text string from the file buffer safely
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    file_content = stringio.read()
    
    # Split text chunks precisely on the established boundary tags
    log_blocks = re.split(r'={5,}\s*INPUT LOG \d+\s*={5,}', file_content)
    
    # Clean up segments
    valid_blocks = []
    for block in log_blocks:
        clean_block = block.strip()
        clean_block = re.sub(r'^={5,}\s*|={5,}\s*$', '', clean_block).strip()
        if clean_block:
            valid_blocks.append(clean_block)
            
    total_logs = len(valid_blocks)
    st.success(f"📊 Successfully loaded {total_logs} test cases from your file!")
    
    # Simple Execution Button
    if st.button("🚀 Analyze Logs", type="primary"):
        output_reports = []
        
        # UI progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.subheader("📋 Analysis Reports")
        
        for idx, clean_block in enumerate(valid_blocks, start=1):
            status_text.text(f"Analyzing log {idx} of {total_logs}...")
            progress_bar.progress(idx / total_logs)
            
            # Format prompt context template behind the scenes
            if "### Instruction:" not in clean_block:
                prompt = f"### Instruction:\n{clean_block}\n\n### Diagnostic Report Output:\n"
            else:
                if "### Diagnostic Report Output:" not in clean_block:
                    prompt = f"{clean_block}\n\n### Diagnostic Report Output:\n"
                else:
                    prompt = clean_block
                    
            # Perform AI Inference
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()} 
            
            with torch.no_grad():
                # Locked safely at 150 maximum tokens automatically
                out = model.generate(**inputs, max_new_tokens=150)
                
            response = tokenizer.decode(out[0], skip_special_tokens=True)
            output_reports.append(response)
            
            # Display results neatly as they finish processing
            with st.container():
                st.markdown(f"### Test Case #{idx} Report")
                st.code(response, language="markdown")
                st.divider()
                
        # Clean up progress graphics
        progress_bar.empty()
        status_text.text("🎉 All logs analyzed successfully!")
        
        # Format the master text file for downloading
        final_download_string = "\n\n" + "="*50 + "\n\n".join(output_reports)
        
        # ==========================================
        # 4. Clean Download Button
        # ==========================================
        st.download_button(
            label="📥 Download Diagnostic Reports",
            data=final_download_string,
            file_name="automotive_diagnostic_reports.txt",
            mime="text/plain"
        )
else:
    st.info("💡 Drop your log file above to get started.")
