# 🎫 Customer Support Ticket Classification System

A comprehensive machine learning pipeline for automatically classifying customer support tickets by issue type and urgency level, with intelligent entity extraction capabilities.

## 🎯 Project Overview

This system addresses the challenge of efficiently categorizing and prioritizing customer support tickets using traditional NLP and machine learning techniques. It provides both a command-line interface and an interactive web application for real-time predictions.

## ✨ Features

### 🔧 Core Functionality
- **Multi-Task Classification**: Simultaneous prediction of issue type and urgency level
- **Entity Extraction**: Automatic identification of products, dates, order numbers, and complaint keywords
- **Confidence Scoring**: Prediction confidence levels for better decision-making
- **Batch Processing**: Handle multiple tickets efficiently

### 🤖 Machine Learning Pipeline
- **Text Preprocessing**: Comprehensive cleaning, tokenization, and lemmatization
- **Feature Engineering**: TF-IDF vectorization + additional features (sentiment, text length, etc.)
- **Model Training**: Random Forest classifiers with cross-validation
- **Performance Evaluation**: Detailed metrics and confusion matrices

### 🌐 Interactive Interface
- **Streamlit **: User-friendly interface for real-time predictions
- **Single Ticket Mode**: Individual ticket classification
- **Batch Upload**: CSV file processing for multiple tickets
- **Visualization Dashboard**: Data exploration and model performance charts

## 📊 Dataset Information

- **Size**: 1000 customer support tickets
- **Issue Types**: 7 categories (Billing Problem, Product Defect, Installation Issue, etc.)
- **Urgency Levels**: 3 levels (Low, Medium, High)
- **Products**: 10 different products (SmartWatch V2, PhotoSnap Cam, etc.)
- **Features**: ticket_id, ticket_text, issue_type, urgency_level, product

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have these files in your directory:
   # - ticket_classifier.py
   # - gradio_app.py
   # - tickets_dataset.csv
   # - requirements.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if not already downloaded)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Running the Application

#### Option 1: Command Line Interface
```bash
python ticket_classifier.py
```
This will:
- Load and preprocess the dataset
- Train both classification models
- Generate performance metrics and visualizations
- Run demo predictions on sample tickets

#### Option 2: Web Interface (Recommended)
```bash
python gradio_app.py
```
Then open your browser and navigate to the provided URL (typically `http://localhost:7860`)

## 🎮 Using the Web Interface

### 1. Model Training
- Navigate to the "🤖 Model Training" tab
- Click "🚀 Train Models" to train the classifiers
- Wait for training completion (shows accuracy metrics)

### 2. Single Ticket Prediction
- Go to "🎯 Single Ticket Prediction" tab
- Enter a ticket description in the text area
- Click "🔮 Predict" to get results
- View issue type, urgency level, and extracted entities

### 3. Batch Processing
- Switch to "📊 Batch Processing" tab
- Upload a CSV file with a 'ticket_text' column
- Click "📈 Process Batch" to process all tickets
- Download results as needed

## 📈 Model Performance

The system achieves strong performance across both classification tasks:

### Issue Type Classification
- **Algorithm**: Random Forest with 100 estimators
- **Features**: TF-IDF (5000 features) + sentiment + text statistics
- **Evaluation**: Cross-validation with stratified splits

### Urgency Level Classification
- **Algorithm**: Random Forest with balanced class weights
- **Features**: Same feature set as issue classification
- **Handling**: Addresses class imbalance in urgency levels

### Entity Extraction
- **Products**: Rule-based matching against known product list
- **Dates**: Regex patterns for multiple date formats
- **Order Numbers**: Pattern matching for order IDs (#XXXXX)
- **Complaint Keywords**: Predefined keyword dictionary

## 🔧 Technical Architecture

### Data Preprocessing Pipeline
1. **Text Cleaning**: Lowercase conversion, special character removal
2. **Tokenization**: Word-level tokenization using NLTK
3. **Stopword Removal**: English stopwords filtering
4. **Lemmatization**: Word normalization using WordNet
5. **Feature Extraction**: TF-IDF + additional engineered features

### Feature Engineering
- **TF-IDF Vectors**: Unigrams and bigrams (max 5000 features)
- **Text Statistics**: Length, word count
- **Sentiment Analysis**: Polarity and subjectivity scores
- **Urgency Indicators**: Presence of urgent keywords

### Model Architecture
```
Input Text → Preprocessing → Feature Engineering → Classification Models
                                                 ├── Issue Type Classifier
                                                 └── Urgency Level Classifier

Input Text → Entity Extraction → Structured Output
```

## 📁 Project Structure

```
├── ticket_classifier.py              # Main ML pipeline and training
├── gradio_app.py                     # Web interface application
├── test_pipeline.py                  # Comprehensive testing suite
├── tickets_dataset.csv               # Training dataset (1000 tickets)
├── sample_tickets.csv                # Sample data for batch testing
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── README.md                         # This documentation
├── DEMO_SCRIPT.md                    # Video recording guide
├── PROJECT_SUMMARY.md                # Technical project summary
└── Generated Visualizations:
    ├── ticket_analysis_dashboard.png
    ├── issue_type_confusion_matrix.png
    └── urgency_confusion_matrix.png
```

## 🎯 Key Design Choices

### **1. Multi-Task Learning Architecture**
```python
# Separate models for different tasks
self.issue_type_model = RandomForestClassifier(...)  # 94.1% accuracy
self.urgency_model = RandomForestClassifier(...)     # 37.9% accuracy
```
**Rationale**: Independent optimization, flexible handling of missing labels, robust failure isolation

### **2. Algorithm Selection: Random Forest**
```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
```
**Why Random Forest**:
- ✅ Excellent baseline performance (94.1% vs 89.2% Logistic Regression)
- ✅ Built-in feature importance analysis
- ✅ Robust to overfitting through ensemble approach
- ✅ Handles mixed data types (text + numerical features)

### **3. Hybrid Feature Engineering**
```python
# Primary: TF-IDF text features
TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Secondary: Engineered features
- text_length, word_count
- sentiment_polarity, sentiment_subjectivity
- has_urgent_words (domain knowledge)
```
**Strategy**: Combine proven NLP techniques with domain-specific features

### **4. Rule-Based Entity Extraction**
```python
# High precision approach
for product in self.products:
    if product.lower() in text.lower():
        entities['products'].append(product)
```
**Trade-off**: 100% precision on known patterns vs. potential recall limitations

### **5. Comprehensive Preprocessing Pipeline**
```python
# Preserve meaning while reducing noise
text → lowercase → remove special chars → tokenize →
remove stopwords → lemmatize → rejoin
```
**Design**: Balance between noise reduction and information preservation

## 🔍 Model Evaluation & Performance

### **Issue Type Classification Results**
```
✅ Overall Accuracy: 94.1%
✅ Precision: 94% (macro avg)
✅ Recall: 94% (macro avg)
✅ F1-Score: 94% (macro avg)
✅ Cross-validation: 94.1% ± 2.1%
```

**Per-Class Performance:**
- Account Access: 96% precision, 94% recall
- Billing Problem: 93% precision, 96% recall
- Product Defect: 96% precision, 95% recall
- All classes achieve >92% performance

### **Urgency Level Classification Results**
```
⚠️ Overall Accuracy: 37.9%
⚠️ Performance close to random baseline (33.3%)
```

**Root Cause Analysis:**
- **Data Quality Issues**: Inconsistent labeling patterns
- **Examples**: "urgent" tickets labeled as Low priority
- **Business Logic Gap**: No consistent urgency assignment rules

### **Entity Extraction Performance**
```
✅ Products: ~100% precision on known products
✅ Order Numbers: ~100% precision (#XXXXX pattern)
✅ Dates: ~90% recall on multiple formats
✅ Keywords: ~85% precision on complaint terms
```

### **Key Metrics Used**
- **Accuracy**: Overall classification performance
- **Precision/Recall/F1**: Per-class performance analysis
- **Confusion Matrix**: Detailed error pattern analysis
- **Cross-Validation**: 5-fold stratified validation for robustness
- **Feature Importance**: TF-IDF and engineered feature analysis

## 🚧 Limitations & Future Improvements

### **Current Limitations**

#### **1. Data Quality Issues (Primary Concern)**
```
❌ Urgency Labeling Inconsistencies:
- "Payment problem. Need urgent help." → Low urgency
- "Can you tell me about warranty?" → High urgency
- Identical texts with different urgency labels
```
**Impact**: 37.9% urgency accuracy reflects data quality, not model capability

#### **2. Entity Extraction Constraints**
```python
# Current: Simple string matching
"SmartWatch V2" ✅ detected
"Smart Watch"   ❌ missed (space variation)
"SW V2"         ❌ missed (abbreviation)
```
**Limitation**: ~15-20% recall loss due to variations and misspellings

#### **3. Scalability Constraints**
- **Memory**: TF-IDF matrix grows with vocabulary (current: 5000 features)
- **Training Time**: O(n log n) complexity for Random Forest
- **Real-time**: 200ms prediction time (acceptable for current scale)

#### **4. Domain Adaptation**
- **Fixed Product List**: Manual updates required for new products
- **English Only**: No multi-language support
- **Customer Support Context**: Not generalizable to other domains

### **Improvement Roadmap**

#### **Short-term (1-2 weeks)**
- ✅ **Data Cleaning**: Rule-based urgency corrections
- ✅ **Enhanced Features**: More text-based urgency indicators
- ✅ **Ensemble Methods**: Combine multiple algorithms

#### **Medium-term (1-2 months)**
- 🔄 **Advanced NER**: spaCy-based entity extraction
- 🔄 **Deep Learning**: BERT/RoBERTa for text understanding
- 🔄 **Active Learning**: User feedback incorporation

#### **Long-term (3-6 months)**
- 📋 **Real-time Learning**: Online model updates
- 📋 **Multi-language**: Support for non-English tickets
- 📋 **Hierarchical Classification**: Issue type → urgency mapping

### **Performance Benchmarks**
```
Current vs Industry Standards:
✅ Issue Classification: 94.1% (Industry: 85-95%)
⚠️ Urgency Prediction: 37.9% (Industry: 70-85%)
✅ Entity Extraction: ~85% (Industry: 80-90%)
✅ Processing Speed: 200ms (Industry: <500ms)
```

## 🤝 Contributing

This project was developed as part of an AI development assignment. For improvements or modifications:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed for educational and demonstration purposes.

## 🙋‍♂️ Support

For questions or issues:
- Check the troubleshooting section below
- Review the code documentation
- Test with the provided example tickets

## 🔧 Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('all')
   ```

2. **Memory Issues with Large Datasets**
   - Reduce TF-IDF max_features parameter
   - Process data in smaller batches

3. **Gradio Interface Not Loading**
   - Check if port 7860 is available
   - Try different port: `interface.launch(server_port=8080)`

4. **Poor Prediction Performance**
   - Ensure model is trained before predictions
   - Check input text quality and length
   - Verify dataset integrity

---

**Built with ❤️ using Python, scikit-learn, NLTK, and Gradio, **
**Made by Batchu Gnana Sampath 😉**
