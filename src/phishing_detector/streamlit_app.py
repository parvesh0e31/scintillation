import os
import streamlit as st
try:
    from .predict import load_model, predict_url
except ImportError:  # supports `streamlit run src/phishing_detector/streamlit_app.py`
    from phishing_detector.predict import load_model, predict_url

st.set_page_config(page_title="URL Phishing Detector")
st.title("URL-only phishing detector")
st.caption("Analysis is local and never fetches the URL.")
url = st.text_input("URL", max_chars=2048, placeholder="https://example.com/login")
if st.button("Analyze") and url:
    try:
        model = load_model(os.getenv("PHISHING_MODEL", "artifacts/model.joblib"))
        result = predict_url(model, url)
        st.metric("Prediction", result["label"])
        st.write(f"Phishing probability: **{result['phishing_probability']:.1%}**")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
