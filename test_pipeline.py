import sys
import os
import pandas as pd
from ticket_classifier import TicketClassifier

def test_data_loading():
    """Test if the dataset can be loaded properly."""
    print("🧪 Testing data loading...")
    
    try:
        classifier = TicketClassifier()
        df = classifier.load_data('tickets_dataset.csv')
        
        assert df is not None, "Dataset is None"
        assert len(df) > 0, "Dataset is empty"
        assert 'ticket_text' in df.columns, "Missing ticket_text column"
        assert 'issue_type' in df.columns, "Missing issue_type column"
        assert 'urgency_level' in df.columns, "Missing urgency_level column"
        
        print("✅ Data loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_text_preprocessing():
    """Test text preprocessing functionality."""
    print("🧪 Testing text preprocessing...")
    
    try:
        classifier = TicketClassifier()
        
        # Test cases
        test_texts = [
            "My SmartWatch V2 is BROKEN!!! Order #12345.",
            "Can you help me with installation???",
            "",  # Empty text
            None  # None value
        ]
        
        for text in test_texts:
            processed = classifier.preprocess_text(text)
            assert isinstance(processed, str), f"Processed text should be string, got {type(processed)}"
        
        print("✅ Text preprocessing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Text preprocessing test failed: {e}")
        return False

def test_entity_extraction():
    """Test entity extraction functionality."""
    print("🧪 Testing entity extraction...")
    
    try:
        classifier = TicketClassifier()
        
        test_text = "My SmartWatch V2 is broken. Order #12345 placed on 15 March. This is urgent!"
        entities = classifier.extract_entities(test_text)
        
        assert isinstance(entities, dict), "Entities should be a dictionary"
        assert 'products' in entities, "Missing products key"
        assert 'dates' in entities, "Missing dates key"
        assert 'complaint_keywords' in entities, "Missing complaint_keywords key"
        assert 'order_numbers' in entities, "Missing order_numbers key"
        
        # Check if entities were extracted
        assert 'SmartWatch V2' in entities['products'], "Product not extracted"
        assert '#12345' in entities['order_numbers'], "Order number not extracted"
        assert 'broken' in entities['complaint_keywords'], "Complaint keyword not extracted"
        
        print("✅ Entity extraction test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        return False

def test_model_training():
    """Test model training functionality."""
    print("🧪 Testing model training...")
    
    try:
        classifier = TicketClassifier()
        
        # Load and prepare data
        df = classifier.load_data('tickets_dataset.csv')
        processed_df, feature_df = classifier.prepare_data(df)
        
        # Train models
        results = classifier.train_models(processed_df, feature_df)
        
        assert isinstance(results, dict), "Results should be a dictionary"
        assert classifier.issue_type_model is not None, "Issue type model not trained"
        assert classifier.urgency_model is not None, "Urgency model not trained"
        assert classifier.tfidf_vectorizer is not None, "TF-IDF vectorizer not fitted"
        
        print("✅ Model training test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

def test_prediction():
    """Test prediction functionality."""
    print("🧪 Testing prediction...")
    
    try:
        classifier = TicketClassifier()
        
        # Load and prepare data
        df = classifier.load_data('tickets_dataset.csv')
        processed_df, feature_df = classifier.prepare_data(df)
        
        # Train models
        classifier.train_models(processed_df, feature_df)
        
        # Test prediction
        test_text = "My SmartWatch V2 is broken and stopped working. Order #12345. This is urgent!"
        prediction = classifier.predict_ticket(test_text)
        
        assert isinstance(prediction, dict), "Prediction should be a dictionary"
        assert 'issue_type' in prediction, "Missing issue_type in prediction"
        assert 'urgency_level' in prediction, "Missing urgency_level in prediction"
        assert 'entities' in prediction, "Missing entities in prediction"
        
        print("✅ Prediction test passed!")
        print(f"   Issue Type: {prediction['issue_type']}")
        print(f"   Urgency: {prediction['urgency_level']}")
        print(f"   Entities: {prediction['entities']}")
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def test_file_existence():
    """Test if all required files exist."""
    print("🧪 Testing file existence...")
    
    required_files = [
        'tickets_dataset.csv',
        'ticket_classifier.py',
        'gradio_app.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist!")
        return True

def run_all_tests():
    """Run all tests and provide summary."""
    print("🚀 Running Customer Support Ticket Classification Pipeline Tests")
    print("=" * 70)
    
    tests = [
        ("File Existence", test_file_existence),
        ("Data Loading", test_data_loading),
        ("Text Preprocessing", test_text_preprocessing),
        ("Entity Extraction", test_entity_extraction),
        ("Model Training", test_model_training),
        ("Prediction", test_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run 'python ticket_classifier.py' for command-line demo")
        print("   2. Run 'python gradio_app.py' for web interface")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
