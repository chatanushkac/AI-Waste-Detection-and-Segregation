# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import pandas as pd
import cv2
import tempfile
import time
import ollama


# ==========================================
# Waste Disposal Recommendation Dictionary
# ==========================================

waste_guidance = {

    "BIODEGRADABLE": {
        "bin": "🟢 Compost Bin",
        "tip": "Biodegradable waste can be composted and converted into organic fertilizer."
    },

    "CARDBOARD": {
        "bin": "📦 Cardboard Recycling Bin",
        "tip": "Cardboard is recyclable and should be kept dry before disposal."
    },

    "GLASS": {
        "bin": "🍾 Glass Recycling Container",
        "tip": "Glass can be recycled multiple times without losing quality."
    },

    "METAL": {
        "bin": "🔩 Metal Recycling Bin",
        "tip": "Metal is one of the most recyclable materials and can be recycled repeatedly."
    },

    "PAPER": {
        "bin": "📄 Paper Recycling Bin",
        "tip": "Paper recycling helps save trees, water, and energy."
    },

    "PLASTIC": {
        "bin": "♻️ Plastic Recycling Bin",
        "tip": "Plastic should be recycled properly to reduce environmental pollution."
    }

}

# Setting page layout
st.set_page_config(
    page_title="Waste Classification using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_llm_analysis(waste_count):

    waste_info = "\n".join(
        [f"{k}: {v}" for k, v in waste_count.items()]
    )

    prompt = f"""
You are an AI Waste Management Expert.

The following waste objects were detected:

{waste_info}

Analyze the detected waste and provide your response in the following format:

<h2 style="color:#166534;">AI Waste Recommendation</h2>

For each detected waste type, provide:
- Correct disposal method
- Recommended bin
- Recycling/handling instructions
- Why this method is recommended.

Write 2-3 sentences for each waste type.

<h2 style="color:#166534;">Contamination Analysis</h2>

- Check whether recyclable and biodegradable waste are mixed.
- Explain if contamination risk exists.

<h2 style="color:#166534;">Environmental Advice</h2>

- Give 2-3 concise suggestions to improve waste management.

Important:
- Do not add unnecessary blank lines.
- Keep each point consecutive.
- Maintain compact formatting.
- Use only one line gap between sections.
"""

    llm_start = time.perf_counter()

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    llm_end = time.perf_counter()

    llm_response_time = llm_end - llm_start

    analysis = response["message"]["content"]

    return analysis, llm_response_time






st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color:  #F1F5F9;
}

[data-testid="stImageCaption"] {
    color: black !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

[data-testid="stAlert"] {
    color: #111827 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #14532D;
}

.main p {
    color: black !important;
}

/* Main Heading */
h1 {
    color: #166534 !important;
    font-weight: 800;
}

/* Subtitle */
h3 {
    color: #15803D !important;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}
div[data-testid="stRadio"] label p {
    color: black !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    font-weight: bold;
    background-color: #22C55E;
    color: white;
}

label, .stRadio label, .stSelectbox label {
    color: black !important;
    font-weight: 600 !important;
}



/* Success & Info Boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
}


/* Download Button Text */
.stDownloadButton > button {
    background-color: #22C55E !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}

/* Waste Summary Header Text */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: white !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

/* ---------- Streamlit 1.58 File Uploader ---------- */

[data-testid="stFileUploader"]{
    background:#0F172A !important;
    border-radius:12px !important;
    padding:10px !important;
    border:none !important;
}

[data-testid="stFileUploader"] section{
    background:#0F172A !important;
    border:none !important;
}

[data-testid="stFileUploader"] small{
    color:white !important;
}

[data-testid="stFileUploader"] label{
    color:white !important;
}

[data-testid="stFileUploader"] button{
    background:white !important;
    color:black !important;
    border-radius:8px !important;
}

[data-testid="stFileUploader"] svg{
    color:white !important;
}

[data-testid="stFileUploader"] section button {
    background-color: #1E293B !important;
    color: white !important;
    opacity: 1 !important;
} 

</style>
""", unsafe_allow_html=True)


# Main page heading
st.markdown("""# ♻️ Smart Waste Classification System""")

st.markdown(
    "<h3 style='color:#94A3B8;'>AI-Powered Waste Detection & Analytics using YOLOv8</h3>",
    unsafe_allow_html=True
)


st.info(
    "Upload a waste image to automatically detect, classify, and analyze waste objects using YOLOv8."
)

# Sidebar
st.sidebar.header("⚙️ Detection Settings")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("📂 Input Configuration")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(
                    default_image_path,
                    caption="📷 Input Image",
                    width="stretch"
                    )
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(
                    source_img,
                    caption="📷 Input Image",
                    width="stretch"
                    )
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(
                default_detected_image_path,
                caption="🤖 AI Detection Output",
                width="stretch"
                )
        else:
            if st.sidebar.button('Detect Objects'):

                start_time = time.time()

                res = model.predict(
                    uploaded_image,
                    conf=confidence
                )

                end_time = time.time()

                inference_time = end_time - start_time

                boxes = res[0].boxes
                confidences = []

                for box in boxes:
                    confidences.append(float(box.conf[0]))

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                res_plotted = res[0].plot()[:, :, ::-1]
                
                st.image(
                    res_plotted,
                    caption="🤖 AI Detection Output",
                    width="stretch"
                )
               
                 # Save detected image temporarily
                temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".jpg"
                        )

                cv2.imwrite(
                    temp_file.name,
                    cv2.cvtColor(
                        res_plotted,
                        cv2.COLOR_RGB2BGR
                    )
                )
                
                
                
                #download button
                with open(temp_file.name, "rb") as file:
                    st.download_button(
                        label="📥 Download Detected Image",
                        data=file,
                        file_name="detected_waste.jpg",
                        mime="image/jpeg"
                    )
    try:
               
        st.markdown(
                    f"""
                    <div style="
                        background-color:white;
                        padding:15px;
                        border-radius:10px;
                        border-left:5px solid green;
                        color:black;
                        font-size:20px;
                        font-weight:bold;">
                        ⚡ Inference Time: {inference_time:.2f} sec
                    </div>
                    """,
                    unsafe_allow_html=True
        )
                
        st.markdown(
                    f"""
                    <div style="
                        background-color:white;
                        padding:15px;
                        border-radius:10px;
                        border-left:5px solid #2563EB;
                        color:black;
                        font-size:20px;
                        font-weight:bold;
                        margin-top:10px;">
                        🎯 Average Confidence: {avg_confidence*100:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        

        class_names = model.names
        waste_count = {}

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]

            if class_name in waste_count:
                waste_count[class_name] += 1
            else:
                waste_count[class_name] = 1

        total_objects = sum(waste_count.values())
        most_detected = max(waste_count, key=waste_count.get)
        
        analysis,llm_response_time = get_llm_analysis(waste_count)

        
        st.markdown(
    f"""
    <div style="
        background:white;
        padding:18px;
        border-radius:12px;
        border-left:6px solid #9333EA;
        color:black;
        font-size:20px;
        font-weight:bold;
        margin-top:15px;
        margin-bottom:15px;">
        🤖 LLM Response Time : {llm_response_time:.3f} sec
    </div>
    """,
    unsafe_allow_html=True
)
        
        st.markdown(
                    "<h2 style='text-align:center;color:black;'>📊 Waste Summary</h2>",
                    unsafe_allow_html=True
                )

        st.markdown(
    f"""
    <div style="
        background:white;
        color:black;
        padding:20px;
        border-radius:12px;
        border-left:6px solid #16A34A;
        font-size:17px;
        line-height:1.8;
        white-space:pre-wrap;
    ">
    {analysis}
    </div>
    """,
    unsafe_allow_html=True
)
                        
        st.markdown(
                    "<h2 style='color:#111827;'>Detected Waste Objects</h2>",
                        unsafe_allow_html=True
                )

        for waste_type, count in waste_count.items():
                    st.markdown(
                        f"<h4 style='color:black'>{waste_type}: {count}</h4>",
                        unsafe_allow_html=True
                    )
                    
        st.markdown(
                    f"""
                    <div style="
                        background-color:white;
                        padding:15px;
                        border-radius:10px;
                        border-left:5px solid #16A34A;
                        color:black;
                        font-size:20px;
                        font-weight:bold;
                        margin-top:10px;">
                        ♻️ Most Detected Waste: {most_detected}
                    </div>
                    """,
                     unsafe_allow_html=True
                ) 
                    
        st.markdown(
                    f"<h2 style='color:#111827;'>Total Objects Detected: {total_objects}</h2>",
                    unsafe_allow_html=True
                )
        
        # ==========================================
        # Waste Contamination Alert System
        # ==========================================

        recyclable_present = False
        biodegradable_present = False

        for waste_type in waste_count.keys():

            if waste_type in [
                "PLASTIC",
                "METAL",
                "GLASS",
                "PAPER",
                "CARDBOARD"
            ]:
                recyclable_present = True

            if waste_type == "BIODEGRADABLE":
                biodegradable_present = True
            
            
        st.markdown(
            "<h2 style='color:#DC2626;'>🚨 Waste Contamination Analysis</h2>",
            unsafe_allow_html=True
        )

        if recyclable_present and biodegradable_present:

            st.error(
                "Mixed recyclable and biodegradable waste detected. Improper segregation may reduce recycling efficiency."
            )

        else:

            st.success(
                "No contamination risk detected. Waste appears properly segregated."
            )
                   
                # ==========================================
                # Waste Disposal Recommendations
                # ==========================================

        st.markdown(
                    "<h2 style='color:#166534;'>♻️ Waste Disposal Recommendations</h2>",
                    unsafe_allow_html=True
                )
        for waste_type in waste_count.keys():

            if waste_type in waste_guidance:

                st.success(
                            f"{waste_type} → {waste_guidance[waste_type]['bin']}"
                        )

                st.info(
                            f"🌱 {waste_guidance[waste_type]['tip']}"
                        )        

        chart_data = pd.DataFrame(
                    waste_count.items(),
                    columns=["Waste Type", "Count"]
                )

        st.bar_chart(
                    chart_data.set_index("Waste Type"),
                    use_container_width=True
                )
    except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!"
                    )

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")


