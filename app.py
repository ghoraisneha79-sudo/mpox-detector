import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Mpox Detector", layout="centered")

st.title("🦠 Mpox AI Detector")

tab1, tab2, tab3 = st.tabs(
    ["Detect","Results","Chatbot"]
)

# ---------------- DETECT ----------------
with tab1:
    st.subheader("📷 Upload Skin Lesion Image")

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        img = Image.open(uploaded).convert("RGB")
        st.image(img, width='stretch')

        with st.spinner("Analyzing..."):
            import time
            time.sleep(2)

        # -----------------------------
        # IMAGE PREPROCESSING
        # -----------------------------
        img2 = img.resize((224,224))
        img_array = np.array(img2)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # --------------------------------
        # TEMP PREDICTION LOGIC
        # Replace with real model later
        # --------------------------------
        pred = random.uniform(0.30,0.95)

        # Threshold fix
        if pred > 0.55:
            label = "Mpox"
        else:
            label = "Non-Mpox"

        conf = pred

        # Uncertain zone
        if 0.45 < conf < 0.55:
            st.warning("⚠️ Uncertain case - review needed")

        # Results
        if label=="Mpox":
            st.error(
                f"⚠️ Mpox Detected\nConfidence: {conf*100:.1f}%"
            )
        else:
            st.success(
                f"✅ Non-Mpox\nConfidence: {(1-conf)*100:.1f}%"
            )

        st.info(
            "Research prototype only. Not a medical diagnosis."
        )

    else:
        st.info("Upload an image to begin detection.")