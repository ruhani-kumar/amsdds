from __future__ import annotations

import os

import requests
import streamlit as st


BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "http://127.0.0.1:5000",
).rstrip("/")

REQUEST_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def fetch_model_info() -> dict | None:
    """Fetch backend/model status information."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/model-info",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_predict(
    file,
    age: str,
    sex: str,
    localization: str,
    layer2_head: str,
) -> dict | None:
    """Send image + patient metadata to the Flask backend."""

    files = {
        "image": (
            file.name,
            file.getvalue(),
            file.type or "application/octet-stream",
        )
    }

    data = {
        "age": age,
        "sex": sex,
        "localization": localization,
        "layer2_head": layer2_head,
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend: {exc}")
        return None

    try:
        payload = response.json()
    except ValueError:
        st.error(
            f"Backend returned a non-JSON response "
            f"(HTTP {response.status_code})."
        )
        return None

    if response.status_code != 200 or not payload.get("success"):
        st.error(
            payload.get(
                "error",
                f"Prediction failed (HTTP {response.status_code}).",
            )
        )
        return None

    return payload


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------

ROUTE_BANNER = {
    "accepted": (
        "success",
        "Layer 1 prediction accepted",
    ),
    "escalated": (
        "info",
        "Prediction escalated to Layer 2",
    ),
    "escalated_failed": (
        "warning",
        "Layer 2 unavailable — showing Layer 1 prediction",
    ),
    "unknown": (
        "warning",
        "Image could not be confidently recognised",
    ),
}


def render_prediction_block(
    title: str,
    data: dict | None,
) -> None:
    """Render prediction details for one layer."""

    if not data:
        return

    st.subheader(title)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Class",
            data.get("class", data.get("label", "-")),
        )

    with c2:
        confidence = data.get("confidence")

        if confidence is None:
            confidence_text = "-"
        else:
            confidence_text = f"{confidence:.3f}"

        st.metric(
            "Confidence",
            confidence_text,
        )

    with c3:
        entropy = data.get("entropy")

        if entropy is None:
            entropy_text = "-"
        else:
            entropy_text = f"{entropy:.3f}"

        st.metric(
            "Entropy",
            entropy_text,
        )


def render_ood_screening(payload: dict) -> None:
    """Render OOD screening information returned by the backend."""

    ood = payload.get("ood")

    if not ood:
        return

    st.divider()
    st.subheader("OOD Screening")

    is_ood = ood.get("is_ood")

    if is_ood:
        st.warning("Rejected — image is out-of-distribution.")
    else:
        st.success("Passed — image is within the recognised distribution.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "OOD Score",
            f"{ood.get('score', 0):.3f}",
        )

    with c2:
        threshold = ood.get("threshold")

        st.metric(
            "OOD Threshold",
            "-" if threshold is None else f"{threshold:.3f}",
        )

    with c3:
        st.metric(
            "OOD Status",
            "Rejected" if is_ood else "Passed",
        )


def render_unknown_result(payload: dict) -> None:
    """Render the OOD / unknown result."""

    st.warning(
        "The image was rejected as out-of-distribution "
        "and was not sent to Layer 2."
    )

    # -----------------------------------------------------------------------
    # Prediction Result
    # -----------------------------------------------------------------------

    st.header("Prediction Result")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Class", "Unknown")

    with c2:
        st.metric("Confidence", "-")

    with c3:
        st.metric("Entropy", "-")

    with c4:
        st.metric("Source", "OOD Detection")

    # -----------------------------------------------------------------------
    # Risk Assessment
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("Risk Assessment")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Malignant Probability", "-")

    with c2:
        st.metric("Risk Level", "Unknown")

    with c3:
        st.metric("Risk Flag", "No")

    advisory = payload.get("advisory")

    if advisory:
        st.info(f"Advisory: {advisory}")

    # -----------------------------------------------------------------------
    # OOD Screening
    # -----------------------------------------------------------------------

    render_ood_screening(payload)

    # -----------------------------------------------------------------------
    # Raw backend response
    # -----------------------------------------------------------------------

    with st.expander("Raw JSON"):
        st.json(payload)


def render_result(payload: dict) -> None:
    """Render the backend prediction response."""

    # -----------------------------------------------------------------------
    # Detect OOD / unknown response
    # -----------------------------------------------------------------------

    is_unknown = payload.get("unknown", False)

    if is_unknown:
        render_unknown_result(payload)
        return

    # -----------------------------------------------------------------------
    # Normal prediction response
    # -----------------------------------------------------------------------

    routing = payload.get("routing", {})

    route = routing.get("route", "unknown")
    uncertain = routing.get("uncertain", False)

    level, text = ROUTE_BANNER.get(
        route,
        ("info", route),
    )

    # Display routing status.
    getattr(st, level)(text)

    # Mock mode warning.
    meta = payload.get("meta", {})

    if meta.get("mock_mode"):
        st.caption(
            "⚠️ Response produced in development mock mode."
        )

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    st.header("Prediction Result")

    final = payload.get("prediction", {})

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Class",
            final.get("class", "-"),
        )

    with c2:
        confidence = final.get("confidence")

        st.metric(
            "Confidence",
            "-" if confidence is None else f"{confidence:.3f}",
        )

    with c3:
        entropy = final.get("entropy")

        st.metric(
            "Entropy",
            "-" if entropy is None else f"{entropy:.3f}",
        )

    with c4:
        st.metric(
            "Source",
            routing.get("final_source", "-"),
        )

    if uncertain:
        st.warning("Prediction is uncertain.")
    else:
        st.success("Prediction is confident.")

    # -----------------------------------------------------------------------
    # Risk information
    # -----------------------------------------------------------------------

    malignant_probability = payload.get("malignant_probability")
    risk_flag = payload.get("risk_flag")
    risk_level = payload.get("risk_level")

    if (
        malignant_probability is not None
        or risk_flag is not None
        or risk_level is not None
    ):
        st.divider()
        st.subheader("Risk Assessment")

        c1, c2, c3 = st.columns(3)

        with c1:
            if malignant_probability is None:
                value = "-"
            else:
                value = f"{malignant_probability:.3f}"

            st.metric(
                "Malignant Probability",
                value,
            )

        with c2:
            st.metric(
                "Risk Level",
                risk_level or "-",
            )

        with c3:
            st.metric(
                "Risk Flag",
                "Yes" if risk_flag else "No",
            )

    # -----------------------------------------------------------------------
    # Advisory
    # -----------------------------------------------------------------------

    advisory = payload.get("advisory")

    if advisory:
        st.info(f"Advisory: {advisory}")

    # -----------------------------------------------------------------------
    # OOD screening
    # -----------------------------------------------------------------------

    render_ood_screening(payload)

    # -----------------------------------------------------------------------
    # Per-layer results
    # -----------------------------------------------------------------------

    st.divider()
    st.header("Model Results")

    render_prediction_block(
        "Layer 1",
        payload.get("layer1"),
    )

    if route in ("escalated", "escalated_failed"):
        render_prediction_block(
            "Layer 2",
            payload.get("layer2"),
        )

    # -----------------------------------------------------------------------
    # Raw response
    # -----------------------------------------------------------------------

    with st.expander("Raw JSON"):
        st.json(payload)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("Adaptive Multi-Layer Skin Disease Detection")

st.caption(
    "Upload a skin lesion image and enter patient information. "
    "The system uses Layer 1 for rapid classification and automatically "
    "escalates uncertain cases to Layer 2."
)

st.divider()


# ---------------------------------------------------------------------------
# Patient information
# ---------------------------------------------------------------------------

st.header("Patient Information")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.text_input(
        "Age",
        placeholder="Enter age",
    )

with c2:
    sex_display = st.selectbox(
        "Sex",
        [
            "Select sex",
            "Male",
            "Female",
        ],
    )

with c3:
    localization = st.selectbox(
        "Lesion Location",
        [
            "Select location",
            "Back",
            "Face",
            "Chest",
            "Abdomen",
            "Upper extremity",
            "Lower extremity",
            "Scalp",
            "Neck",
            "Hand",
            "Foot",
            "Genital",
            "Unknown",
        ],
    )


# ---------------------------------------------------------------------------
# Layer 2 configuration
# ---------------------------------------------------------------------------

st.subheader("Layer 2 Configuration")

layer2_head = st.selectbox(
    "Layer 2 Head",
    [
        "HAMPAD",
        "HAM",
    ],
    index=0,
    help="HAMPAD is the default Layer 2 head.",
)


# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------

st.header("Upload Image")

uploaded = st.file_uploader(
    "Choose a skin lesion image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tiff",
    ],
)

if uploaded is not None:
    st.image(
        uploaded,
        caption="Input image",
        width = 300,
    )


# ---------------------------------------------------------------------------
# Prediction button
# ---------------------------------------------------------------------------

sex_selected = sex_display != "Select sex"
location_selected = localization != "Select location"

predict_clicked = st.button(
    "Predict",
    type="primary",
    use_container_width=True,
    disabled=(
        uploaded is None
        or not sex_selected
        or not location_selected
    ),
)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if predict_clicked and uploaded is not None:

    # Convert UI values to the exact values expected by the backend.
    sex = sex_display.lower()

    if localization == "Select location":
        localization_value = ""
    else:
        localization_value = localization.lower()

    layer2_head_value = layer2_head.lower()

    with st.spinner("Running the pipeline..."):

        result = call_predict(
            uploaded,
            age,
            sex,
            localization_value,
            layer2_head_value,
        )

    if result is not None:
        st.divider()
        render_result(result)
