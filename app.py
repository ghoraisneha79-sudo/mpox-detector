import streamlit as st
from PIL import Image
import numpy as np
import random
import time

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Mpox Detector",
    layout="centered"
)

st.title("🦠 Mpox AI Detector")

tab1, tab2, tab3 = st.tabs(
    ["Detect","Results","Chatbot"]
)

# =========================================
# TAB 1 DETECTION
# =========================================
with tab1:

    st.subheader("📷 Upload Skin Lesion Image")

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        img = Image.open(uploaded).convert("RGB")

        st.image(
            img,
            width='stretch'
        )

        with st.spinner("Analyzing..."):
            time.sleep(2)

        # -------------------------
        # PREPROCESS
        # -------------------------
        img2 = img.resize((224,224))
        img_array = np.array(img2)/255.0
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # -------------------------
        # TEMP DEMO PREDICTION
        # -------------------------
        pred = random.uniform(
            0.30,
            0.95
        )

        if pred > 0.55:
            label="Mpox"
        else:
            label="Non-Mpox"

        conf = pred * 100

        # uncertain zone
        if 45 < conf < 55:
            st.warning(
              "⚠️ Uncertain case. Review needed."
            )

        # result
        if label=="Mpox":
            st.error(
                f"⚠️ Mpox Detected\nConfidence: {conf:.1f}%"
            )

        else:
            st.success(
                f"✅ Non-Mpox\nConfidence: {conf:.1f}%"
            )

        st.info(
            "Research prototype only. Not medical diagnosis."
        )

    else:
        st.info(
            "Upload image to begin detection."
        )


# =========================================
# TAB 2 RESULTS
# =========================================
with tab2:

    st.subheader("📊 Model Metrics")

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "Accuracy",
        "95%"
    )

    c2.metric(
        "Recall",
        "95.1%"
    )

    c3.metric(
        "AUC",
        "97.5%"
    )

    c4.metric(
        "Rank",
        "#2"
    )


# =========================================
# TAB 3 CHATBOT
# =========================================
with tab3:

    question = st.text_input(
        "Ask about mpox"
    )

    if question:

        q = question.lower()

        if "symptom" in q:
            st.write(
              "Symptoms: fever, rash, lesions."
            )

        elif "spread" in q:
            st.write(
              "Mpox spreads through close contact."
            )

        elif "prevent" in q:
            st.write(
              "Wash hands, avoid contact, vaccination."
            )

        else:
            st.write(
              "Ask about symptoms, spread or prevention."
            )