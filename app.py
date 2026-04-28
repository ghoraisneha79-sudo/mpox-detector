with tab1:
    st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload skin lesion image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            import time
            time.sleep(2)

        # ---------- SIMPLE IMAGE-BASED MOCK LOGIC ----------
        # (placeholder until real VGG19 model plugged in)
        img_arr = np.array(img.resize((224,224)))
        brightness = img_arr.mean()

        # dummy heuristic
        if brightness < 120:
            label = "Mpox"
            conf = 0.91
        else:
            label = "Non-Mpox"
            conf = 0.88
        # -----------------------------------------------

        if label == "Mpox":
            st.markdown(
                f'''
                <div class="result-card mpox">
                ⚠️ Mpox Detected
                <br>
                <small>Confidence: {conf*100:.1f}%</small>
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.warning("Consult a healthcare professional.")
        else:
            st.markdown(
                f'''
                <div class="result-card nonmpox">
                ✅ Non-Mpox
                <br>
                <small>Confidence: {conf*100:.1f}%</small>
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.success("No mpox signs detected.")

    else:
        st.markdown("""
        <div class="upload-box">
            <div style="font-size:2.5rem">📸</div>
            <div style="color:#ccc">
                Upload a skin lesion image
            </div>
        </div>
        """, unsafe_allow_html=True)