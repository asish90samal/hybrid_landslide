# ==============================================================
# HYBRID LANDSLIDE DETECTION SYSTEM
# FINAL CLIENT VERSION
# ==============================================================

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="Hybrid Landslide Detection System",
    layout="wide"
)

# ==============================================================
# TRAIN ML MODEL
# ==============================================================

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

    distance_to_road = np.random.uniform(0, 5, data_size)

    terrain_roughness = np.random.uniform(0, 1000, data_size)

    groundwater_level = np.random.uniform(0, 50, data_size)

    precipitation_intensity = np.random.uniform(0, 100, data_size)

    slope_aspect = np.random.uniform(0, 360, data_size)

    soil_type = np.random.randint(1, 5, data_size)

    # ----------------------------------------------------------
    # SYNTHETIC RISK
    # ----------------------------------------------------------

    risk_score = (
        (slope / 60) * 0.20 +
        (rainfall / 350) * 0.20 +
        soil_moisture * 0.15 +
        (1 - ndvi) * 0.08 +
        (lithology / 5) * 0.10 +
        (seismic_activity / 7) * 0.10 +
        (terrain_roughness / 1000) * 0.07 +
        (groundwater_level / 50) * 0.05 +
        (precipitation_intensity / 100) * 0.05
    )

    landslide = (risk_score > 0.45).astype(int)

    # ----------------------------------------------------------
    # DATAFRAME
    # ----------------------------------------------------------

    df = pd.DataFrame({
        "slope": slope,
        "elevation": elevation,
        "rainfall": rainfall,
        "soil_moisture": soil_moisture,
        "ndvi": ndvi,
        "lithology": lithology,
        "seismic_activity": seismic_activity,
        "distance_to_road": distance_to_road,
        "terrain_roughness": terrain_roughness,
        "groundwater_level": groundwater_level,
        "precipitation_intensity": precipitation_intensity,
        "slope_aspect": slope_aspect,
        "soil_type": soil_type,
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
        n_estimators=250,
        max_depth=14,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# ==============================================================
# VIDEO ANALYZER
# ==============================================================

class CombinedVideoAnalyzer:

    def __init__(self):

        self.prev_gray = None

    def process_frame(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------------
        # MOTION DETECTION
        # ------------------------------------------------------

        if self.prev_gray is None:

            movement = 0.0

        else:

            diff = cv2.absdiff(self.prev_gray, gray)

            movement = np.clip(
                diff.mean() * 4.5,
                0,
                100
            )

        self.prev_gray = gray

        # ------------------------------------------------------
        # DEBRIS DETECTION
        # ------------------------------------------------------

        edges = cv2.Canny(gray, 100, 200)

        debris_score = min(
            (edges.mean() / 255.0) * 3.5,
            1.0
        )

        # ------------------------------------------------------
        # PANIC SCORE
        # ------------------------------------------------------

        panic_score = min(
            movement / 100.0,
            1.0
        )

        # ------------------------------------------------------
        # FINAL VIDEO RISK
        # ------------------------------------------------------

        video_risk = min(
            (
                panic_score * 0.65 +
                debris_score * 0.55
            ) * 1.4,
            1.0
        )

        return {
            "panic_score": float(panic_score),
            "debris_score": float(debris_score),
            "video_risk": float(video_risk)
        }


# ==============================================================
# VIDEO ANALYSIS
# ==============================================================

def analyze_video(uploaded_video, max_frames=200):

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

            st.error("Could not open video")

            return None

        analyzer = CombinedVideoAnalyzer()

        panic_scores = []
        debris_scores = []
        video_risks = []

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

            video_risks.append(metrics["video_risk"])

        cap.release()

        if len(video_risks) == 0:
            return None

        return {
            "panic_score": float(np.mean(panic_scores)),
            "debris_score": float(np.mean(debris_scores)),
            "video_risk": float(np.mean(video_risks)),
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


# ==============================================================
# HYBRID SYSTEM
# ==============================================================

class HybridLandslideSystem:

    def __init__(self, model):

        self.model = model

    def predict(self, environmental_data, video_data=None):

        env_df = pd.DataFrame([environmental_data])

        ml_risk = self.model.predict_proba(env_df)[0][1]

        combined_risk = ml_risk

        if video_data is not None:

            combined_risk = (
                ml_risk * 0.80 +
                video_data["video_risk"] * 0.20
            )

        # ------------------------------------------------------
        # ALERT LEVELS
        # ------------------------------------------------------

        if combined_risk > 0.75:

            alert = "🔴 CRITICAL ALERT"

            recommendation = "EVACUATE IMMEDIATELY"

        elif combined_risk > 0.65:

            alert = "🟠 HIGH ALERT"

            recommendation = "PREPARE FOR EVACUATION"

        elif combined_risk > 0.35:

            alert = "🟡 MODERATE ALERT"

            recommendation = "CONTINUE MONITORING"

        else:

            alert = "🟢 LOW RISK"

            recommendation = "SAFE"

        return {
            "ml_risk": float(ml_risk),
            "combined_risk": float(combined_risk),
            "alert": alert,
            "recommendation": recommendation,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# ==============================================================
# MAIN APP
# ==============================================================

def main():

    st.title("🌋 Hybrid Landslide Detection System")

    st.markdown("""
### Features
- Environmental ML Risk Prediction
- Cattle Panic Detection
- Debris Movement Detection
- Hybrid Risk Fusion
- Single Video Analysis
- Synthetic Dataset Fallback
""")

    # ----------------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------------

    with st.spinner("Loading ML model..."):

        model = train_ml_model()

    system = HybridLandslideSystem(model)

    # ----------------------------------------------------------
    # SIDEBAR
    # ----------------------------------------------------------

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
        180.0
    )

    soil_moisture = st.sidebar.slider(
        "Soil Moisture",
        0.1,
        1.0,
        0.6
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

    distance_to_road = st.sidebar.slider(
        "Distance To Road",
        0.0,
        5.0,
        1.5
    )

    terrain_roughness = st.sidebar.slider(
        "Terrain Roughness",
        0.0,
        1000.0,
        500.0
    )

    groundwater_level = st.sidebar.slider(
        "Groundwater Level",
        0.0,
        50.0,
        20.0
    )

    precipitation_intensity = st.sidebar.slider(
        "Precipitation Intensity",
        0.0,
        100.0,
        50.0
    )

    slope_aspect = st.sidebar.slider(
        "Slope Aspect",
        0.0,
        360.0,
        180.0
    )

    soil_type = st.sidebar.selectbox(
        "Soil Type",
        ["Clay", "Sandy", "Loamy", "Rocky"]
    )

    # ----------------------------------------------------------
    # CONVERT SOIL TYPE
    # ----------------------------------------------------------

    soil_map = {
        "Clay": 1,
        "Sandy": 2,
        "Loamy": 3,
        "Rocky": 4
    }

    environmental_data = {
        "slope": slope,
        "elevation": elevation,
        "rainfall": rainfall,
        "soil_moisture": soil_moisture,
        "ndvi": ndvi,
        "lithology": lithology,
        "seismic_activity": seismic_activity,
        "distance_to_road": distance_to_road,
        "terrain_roughness": terrain_roughness,
        "groundwater_level": groundwater_level,
        "precipitation_intensity": precipitation_intensity,
        "slope_aspect": slope_aspect,
        "soil_type": soil_map[soil_type]
    }

    # ----------------------------------------------------------
    # VIDEO INPUT
    # ----------------------------------------------------------

    st.subheader("📹 Upload Video")

    uploaded_video = st.file_uploader(
        "Upload cattle/debris movement video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

    # ----------------------------------------------------------
    # RUN ANALYSIS
    # ----------------------------------------------------------

    if st.button("🚀 Run Analysis"):

        with st.spinner("Running analysis..."):

            video_data = analyze_video(uploaded_video)

            result = system.predict(
                environmental_data,
                video_data
            )

        st.success("Analysis Complete")

        # ------------------------------------------------------
        # METRICS
        # ------------------------------------------------------

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
            f"## Recommendation: {result['recommendation']}"
        )

        st.markdown(
            f"### Timestamp: {result['timestamp']}"
        )

        # ------------------------------------------------------
        # VIDEO SUMMARY
        # ------------------------------------------------------

        if video_data is not None:

            st.subheader("📊 Video Analysis Summary")

            st.json(video_data)

        else:

            st.info(
                "No video uploaded. Using only ML prediction."
            )


# ==============================================================
# RUN
# ==============================================================

if __name__ == "__main__":

    main()