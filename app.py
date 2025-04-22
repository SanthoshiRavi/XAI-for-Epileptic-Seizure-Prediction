import os
import numpy as np
import torch  # This was missing
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from utils.preprocessing import read_mat_file, generate_spectrogram
from utils.explainers import load_model, generate_gradcam, generate_lime_explanation, generate_shap_explanation, generate_integrated_gradients, generate_deeplift_style_ig
from fpdf import FPDF
from datetime import datetime
from fpdf import FPDF, XPos, YPos
from datetime import datetime
import base64


from fpdf import FPDF, XPos, YPos
from datetime import datetime
import os

def create_pdf_report(method_paths, output_dir, signal_length=None, sample_rate=200):
    """Generate a comprehensive clinical PDF report with consistent interpretations"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 15, 'EEG Spectral Analysis Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
              new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(20)
    
    # Patient Information
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Patient Information", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(40, 10, "Patient ID:", new_x=XPos.LMARGIN, new_y=YPos.TOP)
    pdf.cell(0, 10, "_________________________", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.cell(40, 10, "Age/Sex:", new_x=XPos.LMARGIN, new_y=YPos.TOP)
    pdf.cell(0, 10, "_________________________", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.cell(40, 10, "Clinical Notes:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 10, "________________________________________________________________", 
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    
    # Recording Parameters
    duration = signal_length/sample_rate if signal_length else 0
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Recording Parameters", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Duration: {duration:.2f} seconds | Sample Rate: {sample_rate} Hz", 
              new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Frequency Band Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Frequency Bands:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    bands = [
        ["Delta", "0.5-4 Hz", "Deep sleep, pathological states"],
        ["Theta", "4-8 Hz", "Drowsiness, meditation"],
        ["Alpha", "8-12 Hz", "Relaxed wakefulness"],
        ["Beta", "12-30 Hz", "Active thinking, focus"],
        ["Gamma", "30-100 Hz", "Cognitive processing"]
    ]
    
    col_widths = [40, 30, 120]
    row_height = 8  # Slightly reduced for better fit
    
    # Header
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(col_widths[0], row_height, "Band", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(col_widths[1], row_height, "Range", border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.multi_cell(col_widths[2], row_height, "Clinical Significance", border=1, 
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Rows
    pdf.set_font("Helvetica", "", 10)
    for band in bands:
        pdf.cell(col_widths[0], row_height, band[0], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[1], row_height, band[1], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.multi_cell(col_widths[2], row_height, band[2], border=1, 
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(10)
    
    # Spectrogram Analysis
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Spectrogram Analysis', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    if os.path.exists(os.path.join(output_dir, "spectrogram_detailed.png")):
        pdf.image(os.path.join(output_dir, "spectrogram_detailed.png"), x=15, w=180)
    
    # Interpretation
    pdf.set_font('Helvetica', 'I', 10)
    pdf.multi_cell(0, 6, f"""
    Key Observations:
    - X-axis shows time progression (0 to {duration:.1f} seconds)
    - Y-axis displays frequency components (0 to {sample_rate/2:.0f} Hz)
    - Color intensity represents power spectral density (dB)
    - Red/Yellow regions indicate higher energy concentrations
    """, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Method Interpretations
    method_details = {
        "gradcam": {
            "title": "GradCAM Analysis",
            "interpretation": """
            Interpretation:
            - Heatmap shows regions that most influenced the model's decision
            - Red/Yellow = High importance
            - Blue = Low importance
            - Works best for spatial localization
            """
        },
        "lime": {
            "title": "LIME Explanation",
            "interpretation": """
            Interpretation:
            - Green highlights show influential segments
            - Larger areas = More important features
            - Identifies key frequency bands
            """
        },
        "shap": {
            "title": "SHAP Value Analysis",
            "interpretation": """
            Interpretation:
            - Red = Positive impact
            - Blue = Negative impact
            - Intensity shows influence strength
            """
        },
        "integrated_gradients": {
            "title": "Integrated Gradients",
            "interpretation": """
            Interpretation:
            - Blue = Negative influence
            - Red = Positive influence
            - Complete attribution method
            """
        },
        "deeplift": {
            "title": "DeepLIFT Analysis",
            "interpretation": """
            Interpretation:
            - Shows differences from reference
            - Blue-to-Red = Influence direction
            - Handles non-linearities well
            """
        }
    }
    
    for method, path in method_paths.items():
        if os.path.exists(path):
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 10, method_details[method]["title"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            # Center the image
            pdf.image(path, x=(pdf.w - 180)/2, w=180)  # Centered calculation
            
            pdf.set_font('Helvetica', 'I', 10)
            pdf.multi_cell(0, 6, method_details[method]["interpretation"], 
                          new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Clinical Notes Section
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Clinical Correlation Notes', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(0, 8, """
    1. Correlate findings with patient symptoms and history
    2. Review raw EEG traces for verification
    3. Consider medication effects
    4. Note any technical artifacts
    """, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(8)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Physician Notes:', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(0, 10, '__________________________________________________________', 
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.multi_cell(0, 10, '__________________________________________________________', 
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Footer
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 10, 'This report was automatically generated by EEG XAI System. Clinical correlation required.', 
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    
    # Save PDF
    report_path = os.path.join(output_dir, "eeg_clinical_report.pdf")
    pdf.output(report_path)
    return report_path


# Initialize session state to persist method selection
if "selected_method" not in st.session_state:
    st.session_state.selected_method = None

# Configure page
st.set_page_config(
    page_title="EEG XAI App",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader>div>div>div>button {
        background-color: #2196F3;
        color: white;
    }
    .stSelectbox>div>div>div {
        background-color: white;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
        background-image: none;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
    .stMarkdown a {
    background-color: #4CAF50;
    color: white;
    padding: 10px 15px;
    text-decoration: none;
    border-radius: 5px;
    display: inline-block;
    margin-top: 20px;
    }
    .stMarkdown a:hover {
        background-color: #45a049;
    .centered-image {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .interpretation-text {
        text-align: center;
        font-size: 14px;
        color: #555;
        margin-top: 10px;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    }
    </style>
    """, unsafe_allow_html=True)

# Create directories
UPLOAD_DIR = "temp/uploaded_files"
RESULT_DIR = "temp/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# App header
st.title("🧠 EEG Explainability App")
st.markdown("""
    Upload a `.mat` EEG file to visualize the spectrogram and generate model explanations using various XAI methods.
    """)

# === Upload section ===
with st.expander("📁 Upload EEG File", expanded=True):
    uploaded_file = st.file_uploader("Choose a .mat file", type=["mat"], label_visibility="collapsed")

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Read signal and display info
    try:
        signal = read_mat_file(file_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Signal Length", f"{len(signal):,} samples")
        with col2:
            st.metric("Duration", f"{len(signal)/200:.2f} seconds (at 200Hz)")

        # === Spectrogram Preview ===
        st.subheader("📊 Spectrogram Preview")
        with st.spinner("Generating spectrogram..."):
            spectrogram_clean_path = os.path.join(RESULT_DIR, "spectrogram_clean.png")
            spectrogram_detailed_path = os.path.join(RESULT_DIR, "spectrogram_detailed.png")
            
            # Generate spectrograms and get bounds
            spectrogram_bounds = generate_spectrogram(
                signal, 
                spectrogram_clean_path, 
                spectrogram_detailed_path
            )
            
            # Display the detailed spectrogram
            st.image(spectrogram_detailed_path, caption="EEG Spectrogram", use_column_width=True)

        # === Method Selection ===

        st.markdown("---")
        st.subheader("🔍 Explainability Method")
        
        method_descriptions = {
            "GradCAM": "Visualize which regions of the spectrogram influenced the model's decision",
            "SHAP": "Understand feature importance using Shapley values",
            "LIME": "Local interpretable model-agnostic explanations",
            "Integrated Gradients": "Attribute predictions to input features",
            "DeepLift": "Compare activation to a reference input",
            "All": "Run all available methods"
        }
        
        selected_method = st.selectbox(
            "Choose an explainability method",
            options=list(method_descriptions.keys()),
            format_func=lambda x: f"{x} - {method_descriptions[x]}"
        )
        
        st.info(f"ℹ️ {method_descriptions[selected_method]}")

        # === Submit Button ===
        if st.button("🚀 Generate Explanation", use_container_width=True):
            st.info(f"🧠 Processing with: **{selected_method}**")
            
            if selected_method == "GradCAM":
                try:
                    with st.spinner("Generating GradCAM explanation..."):
                        # Load model and generate GradCAM explanation
                        model = load_model(model_path="resnet18_eeg.pt")
                        gradcam_path = generate_gradcam(
                            spectrogram_clean_path, 
                            spectrogram_detailed_path, 
                            model, 
                            RESULT_DIR,
                            spectrogram_bounds
                        )

                    # Display result in two columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(spectrogram_detailed_path, 
                                caption="Original Spectrogram", 
                                use_column_width=True)
                    with col2:
                        st.image(gradcam_path, 
                                caption="GradCAM Explanation", 
                                use_column_width=True)

                    # Interpretation guide
                    st.markdown("""
                    ### 📝 Interpretation Guide
                    - **X-axis**: Time (in seconds)
                    - **Y-axis**: Frequency components (in Hz)
                    - **Color intensity**: Shows which regions influenced the model's classification most
                    - **Red/yellow regions**: Areas where the model focused attention
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error generating GradCAM explanation: {str(e)}")
            elif selected_method == "LIME":
                try:
                    with st.spinner("Generating LIME explanation. This may take 20 - 30 seconds..."):
                        model = load_model(model_path="resnet18_eeg.pt")
                        lime_path = generate_lime_explanation(
                            spectrogram_clean_path,
                            spectrogram_detailed_path,  # Pass detailed image path
                            model,
                            RESULT_DIR
                        )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(spectrogram_detailed_path, 
                            caption="Original Spectrogram", 
                            use_column_width=True)
                    with col2:
                        st.image(lime_path,
                            caption="LIME Explanation", 
                            use_column_width=True)
                        
                    st.markdown("""
                    **Interpretation Guide**:
                    - Green highlighted regions show frequencies/time segments that most influenced the model
                    - Larger green areas indicate more important features
                    - Works best for identifying general important regions rather than precise locations
                    """)
                    
                except Exception as e:
                    st.error(f"LIME failed: {str(e)}")
                    st.info("""
                    Common solutions:
                    1. Try uploading a different EEG sample
                    2. The model might need retraining
                    3. Check console for detailed errors
                    """)
            elif selected_method == "SHAP":
                try:
                    # Simple configuration
                    num_samples = st.number_input(
                        "Number of evaluations",
                        min_value=10,
                        max_value=300,
                        value=100,
                        help="Higher values = more accurate but slower"
                    )

                    with st.spinner(f"🔍 Generating SHAP explanation ({num_samples} samples)..."):
                        # Verify files exist
                        if not os.path.exists(spectrogram_clean_path):
                            raise FileNotFoundError(f"Spectrogram not found at {spectrogram_clean_path}")
                        
                        # Load model with explicit path
                        model = load_model(model_path="resnet18_eeg.pt")  # <-- Add model_path here
                        
                        if not hasattr(model, 'device'):
                            model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model.to(model.device)
                        model.eval()
                        
                        # Generate explanation
                        shap_path = generate_shap_explanation(
                            spectrogram_clean_path,
                            model,
                            RESULT_DIR,
                            num_samples=num_samples
                        )

                        if not os.path.exists(shap_path):
                            raise RuntimeError("SHAP explanation failed to generate")

                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(spectrogram_detailed_path,
                            caption="Original Spectrogram",
                            use_column_width=True)
                    with col2:
                        st.image(shap_path,
                            caption="SHAP Importance Heatmap",
                            use_column_width=True)

                    st.success("""
                    **✅ SHAP Analysis Successful!**  
                    - Red/Yellow = Positive impact  
                    - Blue = Negative impact  
                    - Intensity = Strength of influence
                    """)
                    
                except Exception as e:
                    st.error(f"❌ SHAP Error: {str(e)}")
                    st.info("""
                    Still having trouble? Try:
                    1. Restarting the kernel
                    2. Verifying input is 224x224 RGB
                    3. Updating packages: pip install --upgrade shap numpy torch pillow
                    """)
            elif selected_method == "Integrated Gradients":
                try:
                    with st.spinner("Running Integrated Gradients..."):
                        # Path for clean spectrogram
                        spectrogram_clean_path = os.path.join(RESULT_DIR, "spectrogram_clean.png")
                        spectrogram_detailed_path = os.path.join(RESULT_DIR, "spectrogram_detailed.png")
                        
                        # Generate spectrogram
                        generate_spectrogram(signal, spectrogram_clean_path, spectrogram_detailed_path)
                        
                        # Load model and generate explanation
                        model = load_model(model_path="resnet18_eeg.pt")
                        ig_path = generate_integrated_gradients(spectrogram_clean_path, model, RESULT_DIR)
                    
                    # Display result
                    # In your app.py where you display the image:
                    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
                    with col2:
                        st.image(ig_path, use_column_width=True)
                        st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
                        <h3 style="color: #2c3e50; text-align: center;">Interpretation Guide</h3>
                        <div style="display: flex; justify-content: center; margin: 15px 0;">
                            <div style="width: 200px; height: 20px; background: linear-gradient(to right, #000080, #ffffff, #800000); 
                                        border-radius: 5px; margin: 0 10px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-around; text-align: center;">
                            <div style="color: #000080; font-weight: bold;">Negative Influence</div>
                            <div style="font-weight: bold;">Neutral</div>
                            <div style="color: #800000; font-weight: bold;">Positive Influence</div>
                        </div>
                        <p style="text-align: center; margin-top: 15px;">Color intensity shows relative contribution strength to the model's prediction</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating Integrated Gradients explanation: {str(e)}")
           
            # In your DeepLIFT section:
            elif selected_method == "DeepLift":
                try:
                    with st.spinner("Generating DeepLIFT explanation..."):
                        # Generate spectrograms
                        spectrogram_clean_path = os.path.join(RESULT_DIR, "spectrogram_clean.png")
                        spectrogram_detailed_path = os.path.join(RESULT_DIR, "spectrogram_detailed.png")
                        generate_spectrogram(signal, spectrogram_clean_path, spectrogram_detailed_path)
                        
                        # Load model and generate explanation
                        model = load_model(model_path="resnet18_eeg.pt")
                        explanation_path = generate_deeplift_style_ig(spectrogram_clean_path, model, RESULT_DIR)
                    
                    # Display results
                    st.subheader("DeepLIFT-Style Explanation")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(spectrogram_detailed_path, caption="Original Spectrogram", use_column_width=True)
                    with col2:
                        st.image(explanation_path, caption="Attribution Map", use_column_width=True)
                    
                    # Interpretation guide
                    st.markdown("""
                    ### Interpretation Guide
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 15px;">
                        <div style="display: flex; justify-content: center; margin-bottom: 10px;">
                            <div style="width: 100%; height: 25px; background: linear-gradient(to right, #000080, #ADD8E6, #FFFFFF, #FFA07A, #FF0000); border-radius: 5px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="color: #000080; font-weight: bold;">Negative</span>
                            <span style="font-weight: bold;">Neutral</span>
                            <span style="color: #FF0000; font-weight: bold;">Positive</span>
                        </div>
                        <p style="text-align: center; color: #555;">Color intensity shows feature importance in model's prediction</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Explanation generation failed: {str(e)}")
            
            elif selected_method == "All":
                try:
                    with st.spinner("Generating all explanations (this may take a few minutes)..."):
                        # Load model once
                        model = load_model(model_path="resnet18_eeg.pt")
                        
                        # Generate all explanations
                        method_paths = {}
                        
                        # GradCAM
                        gradcam_path = generate_gradcam(
                            spectrogram_clean_path, 
                            spectrogram_detailed_path, 
                            model, 
                            RESULT_DIR,
                            spectrogram_bounds
                        )
                        method_paths["gradcam"] = gradcam_path
                        
                        # LIME
                        lime_path = generate_lime_explanation(
                            spectrogram_clean_path,
                            spectrogram_detailed_path,
                            model,
                            RESULT_DIR
                        )
                        method_paths["lime"] = lime_path
                        
                        # SHAP
                        shap_path = generate_shap_explanation(
                            spectrogram_clean_path,
                            model,
                            RESULT_DIR,
                            num_samples=100
                        )
                        method_paths["shap"] = shap_path
                        
                        # Integrated Gradients
                        ig_path = generate_integrated_gradients(
                            spectrogram_clean_path,
                            model,
                            RESULT_DIR
                        )
                        method_paths["integrated_gradients"] = ig_path
                        
                        # DeepLift
                        deeplift_path = generate_deeplift_style_ig(
                            spectrogram_clean_path,
                            model,
                            RESULT_DIR
                        )
                        method_paths["deeplift"] = deeplift_path
                        
                        # ====== ADD THIS RIGHT HERE ======
                        # Generate PDF report (with signal length)
                        report_path = create_pdf_report(
                            method_paths=method_paths,
                            output_dir=RESULT_DIR,
                            signal_length=len(signal),  # Pass the signal length
                            sample_rate=200            # And sample rate
                        )
                        # ====== END OF ADDITION ======
                        
                    # Display all results in tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "GradCAM", "LIME", "SHAP", "Integrated Gradients", "DeepLift"
                    ])

                    
                    # Replace the tab display section with this code
                    with tab1:  # GradCAM
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            st.image(gradcam_path, width=500)  # Adjust width as needed
                            st.markdown("""
                            <div style="text-align: center;">
                                <p><b>Interpretation:</b> Heatmap shows regions that most influenced the model's decision.<br>
                                Red/Yellow = High importance | Blue = Low importance</p>
                            </div>
                            """, unsafe_allow_html=True)
                    with tab2:  # LIME
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            st.image(lime_path, width=500)
                            st.markdown("""
                            <div style="text-align: center;">
                                <p><b>Interpretation:</b> Green highlights show influential time-frequency segments.<br>
                                Larger areas = More important features for the classification</p>
                            </div>
                            """, unsafe_allow_html=True)
                    with tab3:  # SHAP
                        st.image(shap_path, use_column_width=True)  # Keep SHAP full width for detail
                        st.markdown("""
                        <div style="text-align: center;">
                            <p><b>Interpretation:</b> Red = Positive impact | Blue = Negative impact<br>
                            Intensity shows strength of influence on the model's prediction</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with tab4:  # Integrated Gradients
                        st.image(ig_path, use_column_width=True)  # Keep full width for multi-panel view
                        st.markdown("""
                        <div style="text-align: center;">
                            <p><b>Interpretation:</b> Blue = Negative influence | Red = Positive influence<br>
                            Complete attribution accounting for baseline activation</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with tab5:  # DeepLIFT
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            st.image(deeplift_path, width=500)
                            st.markdown("""
                            <div style="text-align: center;">
                                <p><b>Interpretation:</b> Shows differences from reference activation<br>
                                Blue-to-Red spectrum indicates direction of influence</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Add download button for PDF report
                    with open(report_path, "rb") as f:
                        pdf_data = f.read()
                    b64 = base64.b64encode(pdf_data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="eeg_analysis_report.pdf">Download Full Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating all explanations: {str(e)}")

    except Exception as e:
        st.error(f"❌ Failed to process .mat file: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9em;">
    EEG Explainability App • Created with Streamlit • 
    <a href="https://github.com/your-repo" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)