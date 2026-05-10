import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(
    page_title="Hybrid Landslide Detection",
    layout="wide"
)

@st.cache_resource
def train_ml_model():

    np.random.seed(42)

    data_size = 5000

    slope = np.random.uniform(0, 60, data_size)
    elevation = np.random.uniform(50, 2500, data_size)
    rainfall = np.random.uniform(10, 350, data_size)
    soil_moisture = np.random.uniform(0.1, 1.0, data_size)
    ndvi = np.random.uniform(-0.2, 0.9, data_size)
    lithology = np.random.randint(1, 6, data_size)
    seismic_activity = np.random.uniform(0, 7, data_size)

    risk_score = (
        (slope / 60) * 0.25 +
        (rainfall / 350) * 0.25 +
        soil_moisture * 0.20 +
        (1 - ndvi) * 0.10 +
        (lithology / 5) * 0.10 +
        (seismic_activity / 7) * 0.10
    )

    landslide = (risk_score > 0.50).astype(int)

    df = pd.DataFrame({
        "slope": slope,
        "elevation": elevation,
        "rainfall": rainfall,
        "soil_moisture": soil_moisture,
        "ndvi": ndvi,
        "lithology": lithology,
        "seismic_activity": seismic_activity,
        "landslide": landslide
    })

    X = df.drop("landslide", axis=1)
    y = df["landslide"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


class CombinedVideoAnalyzer:

    def __init__(self):
        self.prev_gray = None

    def process_frame(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            movement = 0.0
        else:
            diff = cv2.absdiff(self.prev_gray, gray)
            movement = np.clip(diff.mean() * 0.8, 0, 100)

        self.prev_gray = gray

        edges = cv2.Canny(gray, 100, 200)

        debris_score = edges.mean() / 255.0

        panic_score = movement / 100.0

        video_risk = (
            panic_score * 0.55 +
            debris_score * 0.45
        )

        return {
            "panic_score": float(panic_score),
            "debris_score": float(debris_score),
            "video_risk": float(video_risk)
        }


def analyze_video(uploaded_video, max_frames=150):

    if uploaded_video is None:
        return None

    try:

        uploaded_video.seek(0)

        tfile = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        tfile.write(uploaded_video.read())

        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("Could not open uploaded video")
            return None

        analyzer = CombinedVideoAnalyzer()

        panic_scores = []
        debris_scores = []
        total_risks = []

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame_count > max_frames:
                break

            metrics = analyzer.process_frame(frame)

            panic_scores.append(metrics["panic_score"])
            debris_scores.append(metrics["debris_score"])
            total_risks.append(metrics["video_risk"])

        cap.release()

        if len(total_risks) == 0:
            return None

        return {
            "panic_score": float(np.mean(panic_scores)),
            "debris_score": float(np.mean(debris_scores)),
            "video_risk": float(np.mean(total_risks)),
            "frames_analyzed": frame_count
        }

    except Exception as e:

        st.error(f"Video analysis failed: {str(e)}")

        return None

    finally:

        try:
            if 'tfile' in locals():
                os.unlink(tfile.name)
        except:
            pass


class HybridLandslideSystem:

    def __init__(self, model):
        self.model = model

    def predict(self, environmental_data, video_data=None):

        env_df = pd.DataFrame([environmental_data])

        ml_risk = self.model.predict_proba(env_df)[0][1]

        combined_risk = ml_risk

        if video_data is not None:

            combined_risk = (
                ml_risk * 0.75 +
                video_data["video_risk"] * 0.25
            )

        if combined_risk > 0.75:
            alert = "🔴 CRITICAL ALERT"
            recommendation = "EVACUATE IMMEDIATELY"

        elif combined_risk > 0.60:
            alert = "🟠 HIGH ALERT"
            recommendation = "PREPARE FOR EVACUATION"

        elif combined_risk > 0.40:
            alert = "🟡 MODERATE ALERT"
            recommendation = "CONTINUE MONITORING"

        else:
            alert = "🟢 LOW RISK"
            recommendation = "SAFE"

        result = {
            "ml_risk": float(ml_risk),
            "combined_risk": float(combined_risk),
            "alert": alert,
            "recommendation": recommendation,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return result


def main():

    st.title("🌋 Hybrid Landslide Detection System")

    st.markdown("""
### Features
- Environmental ML Prediction
- Cattle Panic Movement Detection
- Small Debris Movement Detection
- Hybrid Risk Assessment
- Single Video Analysis
- Synthetic Dataset Fallback
""")

    with st.spinner("Loading ML model..."):

        model = train_ml_model()

    system = HybridLandslideSystem(model)

    st.sidebar.header("Environmental Parameters")

    slope = st.sidebar.slider(
        "Slope",
        0.0,
        60.0,
        35.0
    )

    elevation = st.sidebar.slider(
        "Elevation",
        50.0,
        2500.0,
        1200.0
    )

    rainfall = st.sidebar.slider(
        "Rainfall",
        10.0,
        350.0,
        150.0
    )

    soil_moisture = st.sidebar.slider(
        "Soil Moisture",
        0.1,
        1.0,
        0.5
    )

    ndvi = st.sidebar.slider(
        "NDVI",
        -0.2,
        0.9,
        0.3
    )

    lithology = st.sidebar.selectbox(
        "Lithology",
        [1, 2, 3, 4, 5]
    )

    seismic_activity = st.sidebar.slider(
        "Seismic Activity",
        0.0,
        7.0,
        3.0
    )

    environmental_data = {
        "slope": slope,
        "elevation": elevation,
        "rainfall": rainfall,
        "soil_moisture": soil_moisture,
        "ndvi": ndvi,
        "lithology": lithology,
        "seismic_activity": seismic_activity
    }

    st.subheader("📹 Upload Video")

    uploaded_video = st.file_uploader(
        "Upload cattle/debris movement video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

    if st.button("🚀 Run Analysis"):

        with st.spinner("Running analysis..."):

            video_data = analyze_video(uploaded_video)

            result = system.predict(
                environmental_data,
                video_data
            )

        st.success("Analysis Complete")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Environmental ML Risk",
                f"{result['ml_risk']:.2f}"
            )

            if video_data is not None:

                st.metric(
                    "Panic Score",
                    f"{video_data['panic_score']:.2f}"
                )

                st.metric(
                    "Debris Score",
                    f"{video_data['debris_score']:.2f}"
                )

        with col2:

            st.metric(
                "Combined Risk",
                f"{result['combined_risk']:.2f}"
            )

            st.metric(
                "Alert Level",
                result["alert"]
            )

        st.markdown(
            f"### Recommendation: {result['recommendation']}"
        )

        st.markdown(
            f"### Timestamp: {result['timestamp']}"
        )

        if video_data is not None:

            st.subheader("📊 Video Analysis Summary")

            st.json(video_data)

        else:

            st.info(
                "No video uploaded. Using only environmental ML prediction."
            )


if __name__ == "__main__":
    main()
