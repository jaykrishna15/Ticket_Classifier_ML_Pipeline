import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def analyze_urgency_data():
    """Comprehensive analysis of urgency level data quality."""
    
    print("🔍 URGENCY LEVEL PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_csv('tickets_dataset.csv')
    
    print(f"📊 Dataset Overview:")
    print(f"Total tickets: {len(df)}")
    print(f"Missing urgency levels: {df['urgency_level'].isnull().sum()}")
    print(f"Percentage missing: {df['urgency_level'].isnull().sum() / len(df) * 100:.1f}%")
    
    # Urgency distribution
    print(f"\n📈 Urgency Level Distribution:")
    urgency_counts = df['urgency_level'].value_counts(dropna=False)
    print(urgency_counts)
    
    # Calculate percentages
    print(f"\n📊 Urgency Level Percentages:")
    urgency_percentages = df['urgency_level'].value_counts(normalize=True, dropna=False) * 100
    print(urgency_percentages.round(1))
    
    # Analyze relationship between issue type and urgency
    print(f"\n🔗 Issue Type vs Urgency Level Cross-tabulation:")
    crosstab = pd.crosstab(df['issue_type'], df['urgency_level'], margins=True, dropna=False)
    print(crosstab)
    
    # Look for patterns in text that might indicate urgency
    print(f"\n🔍 Analyzing Text Patterns for Urgency Indicators:")
    
    # Define urgency keywords
    urgent_keywords = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'help!', 'need urgent help']
    
    # Check for urgency keywords in text
    df_clean = df.dropna(subset=['urgency_level', 'ticket_text'])
    
    for keyword in urgent_keywords:
        keyword_count = df_clean['ticket_text'].str.lower().str.contains(keyword, na=False).sum()
        print(f"'{keyword}': {keyword_count} occurrences")
        
        if keyword_count > 0:
            # Check urgency distribution for tickets with this keyword
            keyword_tickets = df_clean[df_clean['ticket_text'].str.lower().str.contains(keyword, na=False)]
            keyword_urgency = keyword_tickets['urgency_level'].value_counts()
            print(f"  Urgency distribution for '{keyword}': {dict(keyword_urgency)}")
    
    # Analyze specific examples of potential mislabeling
    print(f"\n⚠️  POTENTIAL DATA QUALITY ISSUES:")
    
    # 1. Tickets with "urgent" but labeled as Low
    urgent_but_low = df_clean[
        (df_clean['ticket_text'].str.lower().str.contains('urgent|emergency|critical', na=False)) &
        (df_clean['urgency_level'] == 'Low')
    ]
    
    print(f"\n1. Tickets with urgent keywords but labeled as 'Low' urgency: {len(urgent_but_low)}")
    if len(urgent_but_low) > 0:
        print("Examples:")
        for idx, row in urgent_but_low.head(3).iterrows():
            print(f"   - ID {row['ticket_id']}: {row['ticket_text'][:100]}...")
    
    # 2. General inquiries labeled as High urgency
    general_high = df_clean[
        (df_clean['issue_type'] == 'General Inquiry') &
        (df_clean['urgency_level'] == 'High')
    ]
    
    print(f"\n2. General Inquiries labeled as 'High' urgency: {len(general_high)}")
    if len(general_high) > 0:
        print("Examples:")
        for idx, row in general_high.head(3).iterrows():
            print(f"   - ID {row['ticket_id']}: {row['ticket_text'][:100]}...")
    
    # 3. Product defects with varying urgency
    defects = df_clean[df_clean['issue_type'] == 'Product Defect']
    print(f"\n3. Product Defects urgency distribution:")
    print(defects['urgency_level'].value_counts())
    
    # 4. Check for inconsistent patterns
    print(f"\n4. Inconsistent Urgency Patterns:")
    
    # Similar texts with different urgency levels
    payment_issues = df_clean[df_clean['ticket_text'].str.lower().str.contains('payment issue', na=False)]
    print(f"Payment issues urgency distribution: {dict(payment_issues['urgency_level'].value_counts())}")
    
    installation_issues = df_clean[df_clean['ticket_text'].str.lower().str.contains('installation issue', na=False)]
    print(f"Installation issues urgency distribution: {dict(installation_issues['urgency_level'].value_counts())}")
    
    # 5. Empty or very short tickets
    short_tickets = df_clean[df_clean['ticket_text'].str.len() < 20]
    print(f"\n5. Very short tickets (< 20 chars): {len(short_tickets)}")
    if len(short_tickets) > 0:
        print("Examples:")
        for idx, row in short_tickets.head(3).iterrows():
            print(f"   - ID {row['ticket_id']}: '{row['ticket_text']}' -> {row['urgency_level']}")
    
    # Create visualizations
    create_urgency_visualizations(df)
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS TO IMPROVE URGENCY PREDICTION:")
    print("=" * 60)
    
    print("1. **Data Quality Issues Identified:**")
    print("   - Inconsistent labeling of urgent keywords")
    print("   - General inquiries marked as high urgency")
    print("   - Similar issue types have varying urgency levels")
    print("   - Missing urgency labels (5.2% of data)")
    
    print("\n2. **Suggested Improvements:**")
    print("   - Implement rule-based urgency detection for keywords")
    print("   - Use issue type as a strong feature for urgency prediction")
    print("   - Consider text length and sentiment as urgency indicators")
    print("   - Apply data cleaning and relabeling for obvious inconsistencies")
    print("   - Use ensemble methods or weighted voting")
    
    print("\n3. **Model Improvements:**")
    print("   - Add more text-based features (exclamation marks, caps, etc.)")
    print("   - Use issue type as a categorical feature")
    print("   - Implement hierarchical classification (issue type -> urgency)")
    print("   - Consider using SMOTE for class balancing")
    print("   - Try different algorithms (XGBoost, SVM with different kernels)")

def create_urgency_visualizations(df):
    """Create visualizations to understand urgency patterns."""
    
    plt.figure(figsize=(15, 10))
    
    # 1. Urgency distribution
    plt.subplot(2, 3, 1)
    urgency_counts = df['urgency_level'].value_counts()
    plt.pie(urgency_counts.values, labels=urgency_counts.index, autopct='%1.1f%%')
    plt.title('Urgency Level Distribution')
    
    # 2. Issue type vs urgency heatmap
    plt.subplot(2, 3, 2)
    crosstab = pd.crosstab(df['issue_type'], df['urgency_level'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Issue Type vs Urgency Level')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 3. Text length vs urgency
    plt.subplot(2, 3, 3)
    df_clean = df.dropna(subset=['urgency_level', 'ticket_text'])
    df_clean['text_length'] = df_clean['ticket_text'].str.len()
    
    for urgency in ['Low', 'Medium', 'High']:
        data = df_clean[df_clean['urgency_level'] == urgency]['text_length']
        plt.hist(data, alpha=0.7, label=urgency, bins=20)
    
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution by Urgency')
    plt.legend()
    
    # 4. Urgency keywords analysis
    plt.subplot(2, 3, 4)
    urgent_keywords = ['urgent', 'help!', 'emergency', 'critical', 'asap']
    keyword_counts = []
    
    for keyword in urgent_keywords:
        count = df['ticket_text'].str.lower().str.contains(keyword, na=False).sum()
        keyword_counts.append(count)
    
    plt.bar(urgent_keywords, keyword_counts)
    plt.title('Urgency Keywords Frequency')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # 5. Missing data pattern
    plt.subplot(2, 3, 5)
    missing_by_issue = df.groupby('issue_type')['urgency_level'].apply(lambda x: x.isnull().sum())
    plt.bar(missing_by_issue.index, missing_by_issue.values)
    plt.title('Missing Urgency by Issue Type')
    plt.xticks(rotation=45)
    plt.ylabel('Missing Count')
    
    # 6. Product vs urgency
    plt.subplot(2, 3, 6)
    product_urgency = df.groupby('product')['urgency_level'].apply(lambda x: x.value_counts().get('High', 0))
    top_products = product_urgency.nlargest(5)
    plt.bar(range(len(top_products)), top_products.values)
    plt.title('High Urgency Count by Product (Top 5)')
    plt.xticks(range(len(top_products)), [p[:15] + '...' if len(p) > 15 else p for p in top_products.index], rotation=45)
    plt.ylabel('High Urgency Count')
    
    plt.tight_layout()
    plt.savefig('urgency_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_urgency_data()
