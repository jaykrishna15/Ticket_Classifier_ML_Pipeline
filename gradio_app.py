import gradio as gr
import pandas as pd
import json
from ticket_classifier import TicketClassifier
import warnings
warnings.filterwarnings('ignore')

class TicketClassifierApp:
    """
    Gradio web application for ticket classification.
    """

    def __init__(self):
        """Initialize the app with a pre-trained classifier."""
        self.classifier = None
        self.is_trained = False

    def train_classifier(self):
        """Train the classifier with the dataset."""
        try:
            print("Training classifier...")
            self.classifier = TicketClassifier()

            # Load and prepare data
            df = self.classifier.load_data('tickets_dataset.csv')
            processed_df, feature_df = self.classifier.prepare_data(df)

            # Train models
            results = self.classifier.train_models(processed_df, feature_df)

            self.is_trained = True

            # Return training summary
            summary = "✅ **Training Completed Successfully!**\n\n"

            if 'issue_type' in results:
                summary += f"**Issue Type Classifier Accuracy:** {results['issue_type']['accuracy']:.3f}\n"

            if 'urgency_level' in results:
                summary += f"**Urgency Level Classifier Accuracy:** {results['urgency_level']['accuracy']:.3f}\n"

            summary += "\n🎯 **Ready for predictions!**"

            return summary

        except Exception as e:
            return f"❌ **Training Failed:** {str(e)}"

    def predict_single_ticket(self, ticket_text):
        """
        Predict issue type, urgency, and extract entities for a single ticket.

        Args:
            ticket_text (str): Input ticket text

        Returns:
            tuple: Formatted prediction results
        """
        if not self.is_trained or not self.classifier:
            return (
                "❌ Model not trained. Please train the model first.",
                "❌ Model not trained",
                "❌ Model not trained",
                "❌ Model not trained"
            )

        if not ticket_text.strip():
            return (
                "⚠️ Please enter a ticket description.",
                "No input provided",
                "No input provided",
                "No input provided"
            )

        try:
            # Make prediction
            prediction = self.classifier.predict_ticket(ticket_text)

            # Format results
            issue_type = prediction.get('issue_type', 'Unknown')
            issue_confidence = prediction.get('issue_type_confidence', 0)
            urgency = prediction.get('urgency_level', 'Unknown')
            urgency_confidence = prediction.get('urgency_confidence', 0)
            entities = prediction.get('entities', {})

            # Create formatted output
            summary = f"""
## 🎯 Prediction Results

**Issue Type:** {issue_type} (Confidence: {issue_confidence:.3f})
**Urgency Level:** {urgency} (Confidence: {urgency_confidence:.3f})

### 📋 Extracted Entities:
"""

            for entity_type, entity_list in entities.items():
                if entity_list:
                    summary += f"- **{entity_type.title()}:** {', '.join(entity_list)}\n"
                else:
                    summary += f"- **{entity_type.title()}:** None detected\n"

            # Individual outputs for separate components
            issue_output = f"{issue_type} (Confidence: {issue_confidence:.3f})"
            urgency_output = f"{urgency} (Confidence: {urgency_confidence:.3f})"
            entities_output = json.dumps(entities, indent=2)

            return summary, issue_output, urgency_output, entities_output

        except Exception as e:
            error_msg = f"❌ **Prediction Error:** {str(e)}"
            return error_msg, error_msg, error_msg, error_msg

    def process_batch_tickets(self, file):
        """
        Process multiple tickets from uploaded CSV file.

        Args:
            file: Uploaded CSV file

        Returns:
            tuple: Results dataframe and summary
        """
        if not self.is_trained or not self.classifier:
            return None, "❌ Model not trained. Please train the model first."

        if file is None:
            return None, "⚠️ Please upload a CSV file."

        try:
            # Read uploaded file
            df = pd.read_csv(file)

            if 'ticket_text' not in df.columns:
                return None, "❌ CSV file must contain a 'ticket_text' column."

            # Process each ticket
            results = []
            for idx, row in df.iterrows():
                ticket_text = row['ticket_text']

                if pd.isna(ticket_text) or ticket_text == '':
                    results.append({
                        'ticket_id': idx + 1,
                        'original_text': '',
                        'predicted_issue_type': 'No text provided',
                        'predicted_urgency': 'No text provided',
                        'entities': '{}'
                    })
                    continue

                try:
                    prediction = self.classifier.predict_ticket(str(ticket_text))

                    results.append({
                        'ticket_id': idx + 1,
                        'original_text': ticket_text[:100] + '...' if len(str(ticket_text)) > 100 else ticket_text,
                        'predicted_issue_type': f"{prediction.get('issue_type', 'Unknown')} ({prediction.get('issue_type_confidence', 0):.3f})",
                        'predicted_urgency': f"{prediction.get('urgency_level', 'Unknown')} ({prediction.get('urgency_confidence', 0):.3f})",
                        'entities': json.dumps(prediction.get('entities', {}))
                    })

                except Exception as e:
                    results.append({
                        'ticket_id': idx + 1,
                        'original_text': str(ticket_text)[:100] + '...' if len(str(ticket_text)) > 100 else str(ticket_text),
                        'predicted_issue_type': f'Error: {str(e)}',
                        'predicted_urgency': f'Error: {str(e)}',
                        'entities': '{}'
                    })

            results_df = pd.DataFrame(results)

            summary = f"""
## 📊 Batch Processing Results

**Total Tickets Processed:** {len(results_df)}
**Successfully Processed:** {len([r for r in results if not r['predicted_issue_type'].startswith('Error')])}

### 📈 Issue Type Distribution:
"""

            # Add distribution summary
            issue_types = [r['predicted_issue_type'].split(' (')[0] for r in results if not r['predicted_issue_type'].startswith('Error')]
            if issue_types:
                from collections import Counter
                issue_counts = Counter(issue_types)
                for issue, count in issue_counts.most_common():
                    summary += f"- **{issue}:** {count}\n"

            return results_df, summary

        except Exception as e:
            return None, f"❌ **Batch Processing Error:** {str(e)}"

    def create_interface(self):
        """Create and configure the Gradio interface."""

        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """

        with gr.Blocks(css=css, title="🎫 Ticket Classifier") as interface:

            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🎫 Customer Support Ticket Classifier</h1>
                <p>AI-powered ticket classification and entity extraction system</p>
            </div>
            """)

            # Training Section
            with gr.Tab("🤖 Model Training"):
                gr.Markdown("## Train the Classification Models")
                gr.Markdown("Click the button below to train the models on the ticket dataset.")

                train_btn = gr.Button("🚀 Train Models", variant="primary", size="lg")
                training_output = gr.Markdown()

                train_btn.click(
                    fn=self.train_classifier,
                    outputs=training_output
                )

            # Single Prediction Section
            with gr.Tab("🎯 Single Ticket Prediction"):
                gr.Markdown("## Classify Individual Tickets")

                with gr.Row():
                    with gr.Column(scale=2):
                        ticket_input = gr.Textbox(
                            label="Ticket Description",
                            placeholder="Enter the customer support ticket text here...",
                            lines=5
                        )
                        predict_btn = gr.Button("🔮 Predict", variant="primary")

                    with gr.Column(scale=3):
                        prediction_output = gr.Markdown(label="Prediction Results")

                with gr.Row():
                    issue_output = gr.Textbox(label="Issue Type", interactive=False)
                    urgency_output = gr.Textbox(label="Urgency Level", interactive=False)

                entities_output = gr.Code(label="Extracted Entities (JSON)", language="json")

                predict_btn.click(
                    fn=self.predict_single_ticket,
                    inputs=ticket_input,
                    outputs=[prediction_output, issue_output, urgency_output, entities_output]
                )

                # Example tickets
                gr.Markdown("### 📝 Example Tickets (Click to try)")
                gr.Examples(
                    examples=[
                        ["My SmartWatch V2 is broken and stopped working after 3 days. Order #12345. This is urgent!"],
                        ["Can you tell me more about the warranty for PhotoSnap Cam? Is it available in blue?"],
                        ["Payment issue for order #67890. I was charged twice for my RoboChef Blender."],
                        ["I ordered Vision LED TV but received EcoBreeze AC instead. Order placed on 15 March."],
                        ["Facing installation issue with PowerMax Battery. Setup fails at step 2."]
                    ],
                    inputs=ticket_input
                )

            # Batch Processing Section
            with gr.Tab("📊 Batch Processing"):
                gr.Markdown("## Process Multiple Tickets")
                gr.Markdown("Upload a CSV file with a 'ticket_text' column to process multiple tickets at once.")

                file_input = gr.File(
                    label="Upload CSV File",
                    file_types=[".csv"],
                    type="filepath"
                )

                process_btn = gr.Button("📈 Process Batch", variant="primary")

                batch_summary = gr.Markdown()
                batch_results = gr.Dataframe(
                    label="Processing Results",
                    headers=["Ticket ID", "Original Text", "Predicted Issue Type", "Predicted Urgency", "Entities"]
                )

                process_btn.click(
                    fn=self.process_batch_tickets,
                    inputs=file_input,
                    outputs=[batch_results, batch_summary]
                )

            # Information Section
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## 🎫 Customer Support Ticket Classifier

                This application uses machine learning to automatically classify customer support tickets and extract key information.

                ### 🔧 Features:
                - **Issue Type Classification**: Categorizes tickets into types like Billing Problem, Product Defect, etc.
                - **Urgency Level Prediction**: Determines if a ticket is Low, Medium, or High priority
                - **Entity Extraction**: Identifies products, dates, order numbers, and complaint keywords
                - **Batch Processing**: Handle multiple tickets at once via CSV upload

                ### 🤖 Models Used:
                - **Random Forest Classifier** for both issue type and urgency prediction
                - **TF-IDF Vectorization** for text feature extraction
                - **Rule-based Entity Extraction** for structured information

                ### 📊 Dataset:
                - 1000 customer support tickets
                - 7 issue types and 3 urgency levels
                - 10 different products

                ### 🚀 Getting Started:
                1. Go to the "Model Training" tab and train the models
                2. Use "Single Ticket Prediction" for individual tickets
                3. Use "Batch Processing" for multiple tickets via CSV upload
                """)

        return interface

def main():
    """Launch the Gradio application."""
    app = TicketClassifierApp()
    interface = app.create_interface()

    # Launch the interface
    interface.launch(
        server_name="127.0.0.1",
        server_port=8080,
        share=False,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()
