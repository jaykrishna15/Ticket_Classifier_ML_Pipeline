import streamlit as st
import pandas as pd
import json
from ticket_classifier import TicketClassifier
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "classifier" not in st.session_state:
    st.session_state.classifier = None

if "is_trained" not in st.session_state:
    st.session_state.is_trained = False


# -----------------------------
# TRAIN MODEL FUNCTION
# -----------------------------
def train_classifier():
    try:
        with st.spinner("Training classifier..."):

            classifier = TicketClassifier()

            # Load and prepare data
            df = classifier.load_data("tickets_dataset.csv")
            processed_df, feature_df = classifier.prepare_data(df)

            # Train models
            results = classifier.train_models(processed_df, feature_df)

            st.session_state.classifier = classifier
            st.session_state.is_trained = True

            st.success("Training Completed Successfully!")

            st.subheader("📊 Model Performance")

            if "issue_type" in results:
                st.write(
                    f"**Issue Type Accuracy:** "
                    f"{results['issue_type']['accuracy']:.3f}"
                )

            if "urgency_level" in results:
                st.write(
                    f"**Urgency Level Accuracy:** "
                    f"{results['urgency_level']['accuracy']:.3f}"
                )

    except Exception as e:
        st.error(f"Training Failed: {str(e)}")


# -----------------------------
# SINGLE PREDICTION FUNCTION
# -----------------------------
def predict_ticket(ticket_text):

    if not st.session_state.is_trained:
        st.error("Please train the model first.")
        return

    if not ticket_text.strip():
        st.warning("Please enter ticket text.")
        return

    try:
        prediction = st.session_state.classifier.predict_ticket(ticket_text)

        issue_type = prediction.get("issue_type", "Unknown")
        issue_confidence = prediction.get("issue_type_confidence", 0)

        urgency = prediction.get("urgency_level", "Unknown")
        urgency_confidence = prediction.get("urgency_confidence", 0)

        entities = prediction.get("entities", {})

        # Results
        st.subheader("🎯 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.info(
                f"""
                **Issue Type:**  
                {issue_type}

                **Confidence:**  
                {issue_confidence:.3f}
                """
            )

        with col2:
            st.warning(
                f"""
                **Urgency Level:**  
                {urgency}

                **Confidence:**  
                {urgency_confidence:.3f}
                """
            )

        # Entities
        st.subheader("📋 Extracted Entities")

        st.json(entities)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")


# -----------------------------
# BATCH PROCESS FUNCTION
# -----------------------------
def process_batch(uploaded_file):

    if not st.session_state.is_trained:
        st.error("Please train the model first.")
        return

    try:
        df = pd.read_csv(uploaded_file)

        if "ticket_text" not in df.columns:
            st.error("CSV must contain 'ticket_text' column.")
            return

        results = []

        progress_bar = st.progress(0)

        for idx, row in df.iterrows():

            ticket_text = str(row["ticket_text"])

            prediction = st.session_state.classifier.predict_ticket(ticket_text)

            results.append({
                "Ticket ID": idx + 1,
                "Original Text": ticket_text,
                "Issue Type": prediction.get("issue_type", "Unknown"),
                "Issue Confidence":
                    round(prediction.get("issue_type_confidence", 0), 3),
                "Urgency": prediction.get("urgency_level", "Unknown"),
                "Urgency Confidence":
                    round(prediction.get("urgency_confidence", 0), 3),
                "Entities":
                    json.dumps(prediction.get("entities", {}))
            })

            progress_bar.progress((idx + 1) / len(df))

        results_df = pd.DataFrame(results)

        st.success(f"Processed {len(results_df)} tickets successfully!")

        st.subheader("📊 Results")
        st.dataframe(results_df, use_container_width=True)

        # Distribution
        st.subheader("📈 Issue Type Distribution")

        issue_counts = Counter(results_df["Issue Type"])

        chart_df = pd.DataFrame({
            "Issue Type": list(issue_counts.keys()),
            "Count": list(issue_counts.values())
        })

        st.bar_chart(chart_df.set_index("Issue Type"))

        # Download button
        csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Results CSV",
            data=csv,
            file_name="ticket_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Batch Processing Error: {str(e)}")


# -----------------------------
# UI
# -----------------------------
st.title("🎫 Customer Support Ticket Classifier")
st.markdown(
    "AI-powered ticket classification and entity extraction system"
)

tabs = st.tabs([
    "🤖 Model Training",
    "🎯 Single Prediction",
    "📊 Batch Processing",
    "ℹ️ About"
])

# -----------------------------
# TRAIN TAB
# -----------------------------
with tabs[0]:

    st.header("Train the Classification Models")

    if st.button("🚀 Train Models"):

        train_classifier()

# -----------------------------
# SINGLE PREDICTION TAB
# -----------------------------
with tabs[1]:

    st.header("Classify Individual Tickets")

    ticket_input = st.text_area(
        "Ticket Description",
        placeholder="Enter customer ticket here...",
        height=200
    )

    if st.button("🔮 Predict"):

        predict_ticket(ticket_input)

    st.subheader("📝 Example Tickets")

    examples = [
        "My SmartWatch V2 is broken and stopped working after 3 days.",
        "Payment issue for order #67890. I was charged twice.",
        "Installation issue with PowerMax Battery.",
        "I ordered Vision LED TV but received EcoBreeze AC instead.",
        "Need warranty information for PhotoSnap Cam."
    ]

    for example in examples:
        st.code(example)

# -----------------------------
# BATCH TAB
# -----------------------------
with tabs[2]:

    st.header("Process Multiple Tickets")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        if st.button("📈 Process Batch"):

            process_batch(uploaded_file)

# -----------------------------
# ABOUT TAB
# -----------------------------
with tabs[3]:

    st.header("ℹ️ About")

    st.markdown("""
    ### Features
    - Issue Type Classification
    - Urgency Prediction
    - Entity Extraction
    - Batch CSV Processing

    ### Models Used
    - Random Forest Classifier
    - TF-IDF Vectorization
    - Rule-based Entity Extraction

    ### Dataset
    - 1000 customer support tickets
    - 7 issue types
    - 3 urgency levels

    ### Workflow
    1. Train the model
    2. Predict single tickets
    3. Process CSV files in batch
    """)
