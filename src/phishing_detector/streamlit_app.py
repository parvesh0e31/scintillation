import os
import streamlit as st
try:
    from .predict import load_model, predict_url
    from .features import extract_features
except ImportError:  # supports `streamlit run src/phishing_detector/streamlit_app.py`
    from phishing_detector.predict import load_model, predict_url
    from phishing_detector.features import extract_features


def main() -> None:
    st.set_page_config(page_title="URL Phishing Detector")
    st.title("URL-only phishing detector")
    st.caption(
        "Analysis is local and never fetches the URL. This research baseline "
        "is not a security verdict; its score is not calibrated."
    )
    url = st.text_input("URL", max_chars=2048, placeholder="https://example.com/login")
    if st.button("Analyze") and url:
        try:
            model = load_model(os.getenv("PHISHING_MODEL", "models/model.joblib"))
            result = predict_url(model, url)
            st.metric("Prediction", result["label"])
            st.write(f"Model phishing score: **{result['phishing_probability']:.1%}**")
            with st.expander("Extracted URL features"):
                st.caption("Input signals used by the model, not an explanation of its decision.")
                st.dataframe(
                    [{"Feature": name, "Value": value} for name, value in extract_features(url).items()],
                    hide_index=True,
                    use_container_width=True,
                )
        except (ValueError, OSError) as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
