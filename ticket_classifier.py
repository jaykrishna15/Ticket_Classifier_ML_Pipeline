import pandas as pd
import numpy as np
import re
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

# ML and NLP libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

class TicketClassifier:
    """
    A comprehensive ticket classification and entity extraction system.
    """

    def __init__(self):
        """Initialize the classifier with necessary components."""
        self.issue_type_model = None
        self.urgency_model = None
        self.tfidf_vectorizer = None
        self.issue_type_encoder = None
        self.urgency_encoder = None
        self.lemmatizer = WordNetLemmatizer()

        # Download required NLTK data
        self._download_nltk_data()

        # Define product list and complaint keywords
        self.products = [
            'SmartWatch V2', 'UltraClean Vacuum', 'SoundWave 300', 'PhotoSnap Cam',
            'Vision LED TV', 'RoboChef Blender', 'EcoBreeze AC', 'PowerMax Battery',
            'ProTab X1', 'FitRun Treadmill'
        ]

        self.complaint_keywords = [
            'broken', 'damaged', 'defective', 'faulty', 'malfunction', 'error',
            'late', 'delayed', 'missing', 'lost', 'wrong', 'incorrect', 'blocked',
            'failed', 'not working', 'stopped working', 'glitchy', 'stuck',
            'cracked', 'charged', 'underbilled', 'overbilled', 'refunded'
        ]

    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and perform initial data exploration.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("Loading dataset...")
        df = pd.read_csv(file_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nMissing values:")
        print(df.isnull().sum())

        print("\nIssue type distribution:")
        print(df['issue_type'].value_counts())

        print("\nUrgency level distribution:")
        print(df['urgency_level'].value_counts())

        return df

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data with comprehensive cleaning.

        Args:
            text (str): Raw text to preprocess

        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or text == '':
            return ''

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                 if token not in stop_words and len(token) > 2]

        return ' '.join(tokens)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features from the text data.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        print("Extracting additional features...")

        # Text length features
        df['text_length'] = df['ticket_text'].fillna('').astype(str).apply(len)
        df['word_count'] = df['ticket_text'].fillna('').astype(str).apply(lambda x: len(x.split()))

        # Sentiment analysis
        df['sentiment_polarity'] = df['ticket_text'].fillna('').apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        df['sentiment_subjectivity'] = df['ticket_text'].fillna('').apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity
        )

        # Urgency indicators
        urgent_words = ['urgent', 'asap', 'immediately', 'emergency', 'critical']
        df['has_urgent_words'] = df['ticket_text'].fillna('').astype(str).apply(
            lambda x: any(word in x.lower() for word in urgent_words)
        ).astype(int)

        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training by handling missing values and preprocessing.

        Args:
            df (pd.DataFrame): Raw dataframe

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed dataframe and feature dataframe
        """
        print("Preparing data...")

        # Handle missing values
        df = df.copy()
        df['ticket_text'] = df['ticket_text'].fillna('')

        # Remove rows where both issue_type and urgency_level are missing
        df = df.dropna(subset=['issue_type', 'urgency_level'], how='all')

        # Reset index to ensure continuous indexing
        df = df.reset_index(drop=True)

        # Preprocess text
        df['processed_text'] = df['ticket_text'].apply(self.preprocess_text)

        # Extract additional features
        df = self.extract_features(df)

        # Create feature dataframe
        feature_df = df[['processed_text', 'text_length', 'word_count',
                        'sentiment_polarity', 'sentiment_subjectivity', 'has_urgent_words']].copy()

        return df, feature_df

    def train_models(self, df: pd.DataFrame, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train both issue type and urgency level classification models.

        Args:
            df (pd.DataFrame): Processed dataframe with labels
            feature_df (pd.DataFrame): Feature dataframe

        Returns:
            Dict[str, Any]: Training results and metrics
        """
        print("Training models...")
        results = {}

        # Prepare TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        # Fit TF-IDF on all available text
        text_features = self.tfidf_vectorizer.fit_transform(feature_df['processed_text'])

        # Combine TF-IDF with additional features
        additional_features = feature_df[['text_length', 'word_count',
                                        'sentiment_polarity', 'sentiment_subjectivity',
                                        'has_urgent_words']].values

        # Combine features
        from scipy.sparse import hstack
        X = hstack([text_features, additional_features])
        # Convert to CSR format for better indexing support
        X = X.tocsr()

        # Train Issue Type Classifier
        issue_data = df.dropna(subset=['issue_type'])
        if len(issue_data) > 0:
            y_issue = issue_data['issue_type']
            # Create boolean mask for valid indices
            mask = np.zeros(len(df), dtype=bool)
            mask[issue_data.index] = True
            X_issue = X[mask]

            self.issue_type_encoder = LabelEncoder()
            y_issue_encoded = self.issue_type_encoder.fit_transform(y_issue)

            # Split data
            X_train_issue, X_test_issue, y_train_issue, y_test_issue = train_test_split(
                X_issue, y_issue_encoded, test_size=0.2, random_state=42, stratify=y_issue_encoded
            )

            # Train Random Forest for issue type
            self.issue_type_model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            self.issue_type_model.fit(X_train_issue, y_train_issue)

            # Evaluate
            y_pred_issue = self.issue_type_model.predict(X_test_issue)
            issue_accuracy = accuracy_score(y_test_issue, y_pred_issue)

            results['issue_type'] = {
                'accuracy': issue_accuracy,
                'classification_report': classification_report(
                    y_test_issue, y_pred_issue,
                    target_names=self.issue_type_encoder.classes_
                ),
                'confusion_matrix': confusion_matrix(y_test_issue, y_pred_issue)
            }

            print(f"Issue Type Classifier Accuracy: {issue_accuracy:.3f}")

        # Train Urgency Level Classifier
        urgency_data = df.dropna(subset=['urgency_level'])
        if len(urgency_data) > 0:
            y_urgency = urgency_data['urgency_level']
            # Create boolean mask for valid indices
            mask = np.zeros(len(df), dtype=bool)
            mask[urgency_data.index] = True
            X_urgency = X[mask]

            self.urgency_encoder = LabelEncoder()
            y_urgency_encoded = self.urgency_encoder.fit_transform(y_urgency)

            # Split data
            X_train_urgency, X_test_urgency, y_train_urgency, y_test_urgency = train_test_split(
                X_urgency, y_urgency_encoded, test_size=0.2, random_state=42, stratify=y_urgency_encoded
            )

            # Train Random Forest for urgency
            self.urgency_model = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            self.urgency_model.fit(X_train_urgency, y_train_urgency)

            # Evaluate
            y_pred_urgency = self.urgency_model.predict(X_test_urgency)
            urgency_accuracy = accuracy_score(y_test_urgency, y_pred_urgency)

            results['urgency_level'] = {
                'accuracy': urgency_accuracy,
                'classification_report': classification_report(
                    y_test_urgency, y_pred_urgency,
                    target_names=self.urgency_encoder.classes_
                ),
                'confusion_matrix': confusion_matrix(y_test_urgency, y_pred_urgency)
            }

            print(f"Urgency Level Classifier Accuracy: {urgency_accuracy:.3f}")

        return results

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from ticket text using rule-based methods.

        Args:
            text (str): Input text

        Returns:
            Dict[str, List[str]]: Extracted entities
        """
        if pd.isna(text) or text == '':
            return {'products': [], 'dates': [], 'complaint_keywords': [], 'order_numbers': []}

        text = str(text)
        entities = {
            'products': [],
            'dates': [],
            'complaint_keywords': [],
            'order_numbers': []
        }

        # Extract products
        for product in self.products:
            if product.lower() in text.lower():
                entities['products'].append(product)

        # Extract dates (various formats)
        date_patterns = [
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['dates'].extend(matches)

        # Extract complaint keywords
        for keyword in self.complaint_keywords:
            if keyword.lower() in text.lower():
                entities['complaint_keywords'].append(keyword)

        # Extract order numbers
        order_pattern = r'#\d+'
        order_matches = re.findall(order_pattern, text)
        entities['order_numbers'] = order_matches

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def predict_ticket(self, ticket_text: str) -> Dict[str, Any]:
        """
        Predict issue type, urgency level, and extract entities for a single ticket.

        Args:
            ticket_text (str): Raw ticket text

        Returns:
            Dict[str, Any]: Predictions and extracted entities
        """
        if not self.issue_type_model or not self.urgency_model or not self.tfidf_vectorizer:
            raise ValueError("Models not trained. Please train models first.")

        # Preprocess text
        processed_text = self.preprocess_text(ticket_text)

        # Extract additional features
        text_length = len(ticket_text)
        word_count = len(ticket_text.split())
        sentiment = TextBlob(ticket_text).sentiment
        urgent_words = ['urgent', 'asap', 'immediately', 'emergency', 'critical']
        has_urgent_words = int(any(word in ticket_text.lower() for word in urgent_words))

        # Create feature vector
        text_features = self.tfidf_vectorizer.transform([processed_text])
        additional_features = np.array([[text_length, word_count, sentiment.polarity,
                                       sentiment.subjectivity, has_urgent_words]])

        # Combine features
        from scipy.sparse import hstack
        X = hstack([text_features, additional_features])

        # Make predictions
        issue_type_pred = None
        urgency_pred = None

        if self.issue_type_model and self.issue_type_encoder:
            issue_type_encoded = self.issue_type_model.predict(X)[0]
            issue_type_pred = self.issue_type_encoder.inverse_transform([issue_type_encoded])[0]
            issue_type_proba = self.issue_type_model.predict_proba(X)[0]
            issue_type_confidence = max(issue_type_proba)

        if self.urgency_model and self.urgency_encoder:
            urgency_encoded = self.urgency_model.predict(X)[0]
            urgency_pred = self.urgency_encoder.inverse_transform([urgency_encoded])[0]
            urgency_proba = self.urgency_model.predict_proba(X)[0]
            urgency_confidence = max(urgency_proba)

        # Extract entities
        entities = self.extract_entities(ticket_text)

        return {
            'issue_type': issue_type_pred,
            'issue_type_confidence': issue_type_confidence if issue_type_pred else None,
            'urgency_level': urgency_pred,
            'urgency_confidence': urgency_confidence if urgency_pred else None,
            'entities': entities
        }

    def create_visualizations(self, df: pd.DataFrame, results: Dict[str, Any]) -> None:
        """
        Create visualizations for data exploration and model performance.

        Args:
            df (pd.DataFrame): Dataset
            results (Dict[str, Any]): Training results
        """
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Support Ticket Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Issue Type Distribution
        issue_counts = df['issue_type'].value_counts()
        axes[0, 0].pie(issue_counts.values, labels=issue_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Issue Type Distribution')

        # 2. Urgency Level Distribution
        urgency_counts = df['urgency_level'].value_counts()
        axes[0, 1].bar(urgency_counts.index, urgency_counts.values, color=['red', 'orange', 'green'])
        axes[0, 1].set_title('Urgency Level Distribution')
        axes[0, 1].set_xlabel('Urgency Level')
        axes[0, 1].set_ylabel('Count')

        # 3. Product Distribution
        product_counts = df['product'].value_counts()
        axes[0, 2].barh(product_counts.index, product_counts.values)
        axes[0, 2].set_title('Product Distribution')
        axes[0, 2].set_xlabel('Count')

        # 4. Text Length Distribution
        axes[1, 0].hist(df['text_length'], bins=30, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Ticket Text Length Distribution')
        axes[1, 0].set_xlabel('Text Length (characters)')
        axes[1, 0].set_ylabel('Frequency')

        # 5. Sentiment Analysis
        axes[1, 1].scatter(df['sentiment_polarity'], df['sentiment_subjectivity'], alpha=0.6)
        axes[1, 1].set_title('Sentiment Analysis')
        axes[1, 1].set_xlabel('Polarity (Negative ← → Positive)')
        axes[1, 1].set_ylabel('Subjectivity (Objective ← → Subjective)')

        # 6. Model Performance
        if 'issue_type' in results and 'urgency_level' in results:
            accuracies = [results['issue_type']['accuracy'], results['urgency_level']['accuracy']]
            models = ['Issue Type', 'Urgency Level']
            bars = axes[1, 2].bar(models, accuracies, color=['lightblue', 'lightcoral'])
            axes[1, 2].set_title('Model Accuracy Comparison')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].set_ylim(0, 1)

            # Add accuracy values on bars
            for bar, acc in zip(bars, accuracies):
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('ticket_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Create confusion matrices
        if 'issue_type' in results:
            plt.figure(figsize=(10, 8))
            cm = results['issue_type']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.issue_type_encoder.classes_,
                       yticklabels=self.issue_type_encoder.classes_)
            plt.title('Issue Type Classification - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('issue_type_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()

        if 'urgency_level' in results:
            plt.figure(figsize=(8, 6))
            cm = results['urgency_level']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                       xticklabels=self.urgency_encoder.classes_,
                       yticklabels=self.urgency_encoder.classes_)
            plt.title('Urgency Level Classification - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('urgency_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()


def main():
    """
    Main function to demonstrate the ticket classification pipeline.
    """
    print("🎫 Customer Support Ticket Classification Pipeline")
    print("=" * 60)

    # Initialize classifier
    classifier = TicketClassifier()

    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    df = classifier.load_data('tickets_dataset.csv')
    processed_df, feature_df = classifier.prepare_data(df)

    # Train models
    print("\n🤖 Training models...")
    results = classifier.train_models(processed_df, feature_df)

    # Print detailed results
    print("\n📈 Training Results:")
    print("-" * 40)

    if 'issue_type' in results:
        print(f"Issue Type Classifier:")
        print(f"  Accuracy: {results['issue_type']['accuracy']:.3f}")
        print(f"  Classification Report:")
        print(results['issue_type']['classification_report'])

    if 'urgency_level' in results:
        print(f"\nUrgency Level Classifier:")
        print(f"  Accuracy: {results['urgency_level']['accuracy']:.3f}")
        print(f"  Classification Report:")
        print(results['urgency_level']['classification_report'])

    # Create visualizations
    print("\n📊 Creating visualizations...")
    classifier.create_visualizations(processed_df, results)

    # Demo predictions
    print("\n🔮 Demo Predictions:")
    print("-" * 40)

    demo_tickets = [
        "My SmartWatch V2 is broken and stopped working after 3 days. Order #12345. This is urgent!",
        "Can you tell me more about the warranty for PhotoSnap Cam? Is it available in blue?",
        "Payment issue for order #67890. I was charged twice for my RoboChef Blender.",
        "I ordered Vision LED TV but received EcoBreeze AC instead. Order placed on 15 March.",
        "Facing installation issue with PowerMax Battery. Setup fails at step 2."
    ]

    for i, ticket in enumerate(demo_tickets, 1):
        print(f"\nDemo Ticket {i}:")
        print(f"Text: {ticket}")

        try:
            prediction = classifier.predict_ticket(ticket)
            print(f"Predicted Issue Type: {prediction['issue_type']} (confidence: {prediction['issue_type_confidence']:.3f})")
            print(f"Predicted Urgency: {prediction['urgency_level']} (confidence: {prediction['urgency_confidence']:.3f})")
            print(f"Extracted Entities: {json.dumps(prediction['entities'], indent=2)}")
        except Exception as e:
            print(f"Error in prediction: {e}")

        print("-" * 40)

    print("\n✅ Pipeline execution completed!")
    print("📁 Generated files:")
    print("  - ticket_analysis_dashboard.png")
    print("  - issue_type_confusion_matrix.png")
    print("  - urgency_confusion_matrix.png")

    return classifier


if __name__ == "__main__":
    # Run the main pipeline
    trained_classifier = main()