#!/usr/bin/env python3
"""
Simple Gradio Test to Debug Web Interface Issues
"""

import gradio as gr
import pandas as pd
import json
from ticket_classifier import TicketClassifier

def simple_predict(ticket_text):
    """Simple prediction function for testing."""
    if not ticket_text.strip():
        return "Please enter a ticket description.", "No input", "No input", "{}"
    
    try:
        # Create a simple mock response for testing
        return (
            f"**Test Result for:** {ticket_text[:50]}...\n\n**Status:** Interface working correctly!",
            "Product Defect (Test)",
            "Medium (Test)", 
            '{"products": ["Test Product"], "status": "working"}'
        )
    except Exception as e:
        return f"Error: {str(e)}", "Error", "Error", "{}"

def create_simple_interface():
    """Create a simple test interface."""
    
    with gr.Blocks(title="🎫 Ticket Classifier - Test") as interface:
        
        gr.HTML("""
        <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1>🎫 Customer Support Ticket Classifier - Test Mode</h1>
            <p>Testing web interface functionality</p>
        </div>
        """)
        
        with gr.Tab("🎯 Simple Test"):
            gr.Markdown("## Test the Interface")
            
            with gr.Row():
                with gr.Column():
                    ticket_input = gr.Textbox(
                        label="Ticket Description",
                        placeholder="Enter test ticket text here...",
                        lines=3
                    )
                    predict_btn = gr.Button("🔮 Test Predict", variant="primary")
                
                with gr.Column():
                    result_output = gr.Markdown(label="Test Results")
            
            with gr.Row():
                issue_output = gr.Textbox(label="Issue Type", interactive=False)
                urgency_output = gr.Textbox(label="Urgency Level", interactive=False)
            
            entities_output = gr.Code(label="Entities (JSON)", language="json")
            
            predict_btn.click(
                fn=simple_predict,
                inputs=ticket_input,
                outputs=[result_output, issue_output, urgency_output, entities_output]
            )
            
            # Test examples
            gr.Examples(
                examples=[
                    ["Test ticket: My product is broken"],
                    ["Test ticket: Need help with installation"],
                    ["Test ticket: Payment issue with order"]
                ],
                inputs=ticket_input
            )
        
        with gr.Tab("ℹ️ Debug Info"):
            gr.Markdown("""
            ## 🔧 Debug Information
            
            **Interface Status:** ✅ Running
            
            **Test Steps:**
            1. Enter text in the ticket description
            2. Click "Test Predict" button
            3. Check if results appear
            
            **Expected Behavior:**
            - Text should appear in results
            - Issue and urgency fields should show test values
            - JSON should display in entities section
            
            **If this works:** The basic interface is functional
            **If this fails:** There's a fundamental Gradio issue
            """)
    
    return interface

def main():
    """Launch the simple test interface."""
    print("🧪 Starting Simple Gradio Test...")
    print("=" * 40)
    
    interface = create_simple_interface()
    
    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=8081,
            share=False,
            show_error=True,
            debug=True,
            inbrowser=True  # Automatically open browser
        )
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check if port 8081 is available")
        print("2. Try a different port")
        print("3. Check Gradio installation: pip install --upgrade gradio")

if __name__ == "__main__":
    main()
