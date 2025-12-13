import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nlp_utils import create_tfidf_vectorizer, get_reason_embedding, categorize_reason

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_synthetic_dataset(n_samples=1000):
    """
    Generate synthetic leave request dataset with realistic data.
    """
    
    # Define possible values
    departments = ['Engineering', 'HR', 'Finance', 'Marketing', 'Operations', 'Sales', 'IT', 'Admin']
    employment_statuses = ['Permanent', 'Probation', 'Contract']
    leave_types = ['Sick', 'Casual', 'Earned', 'Emergency']
    
    # Reason templates by category
    reason_templates = {
        'Medical': [
            "I am suffering from high fever and need to visit the doctor for checkup",
            "I have a medical appointment scheduled at the hospital for health issues",
            "Experiencing severe headache and stomach pain requiring medical attention",
            "Need to undergo surgery and recovery time at home",
            "Having flu symptoms and doctor advised rest for few days",
            "Diagnosed with infection and need time for medication and treatment",
            "Dental emergency requiring immediate clinic visit and recovery",
            "Chronic back pain has worsened and need physiotherapy sessions",
            "Eye checkup appointment with specialist ophthalmologist scheduled",
            "Dealing with severe migraine attacks requiring rest and medication"
        ],
        'Emergency': [
            "Family emergency due to sudden hospitalization of parent requiring immediate attention",
            "Urgent situation at home needs immediate presence and resolution",
            "Death in the family and need to attend funeral arrangements",
            "Child met with an accident and needs immediate care",
            "Critical emergency involving elderly family member hospitalization",
            "Unexpected family crisis requiring immediate travel back home",
            "Close relative in critical condition at emergency room",
            "Home emergency due to natural disaster requiring immediate action",
            "Sudden family bereavement and funeral ceremony attendance",
            "Urgent legal matter requiring immediate personal presence"
        ],
        'Personal': [
            "Attending family wedding ceremony and related functions",
            "Child's school annual day function and parent participation required",
            "Moving to new house and need time for relocation activities",
            "Planned family vacation to spend quality time together",
            "Religious festival celebration with family gathering",
            "Anniversary celebration with spouse at planned destination",
            "Attending important family function out of town",
            "Parent's birthday celebration and family get together",
            "Personal commitment to attend cousin's marriage ceremony",
            "Cultural event participation with family members",
            "Home renovation work supervision requiring presence",
            "Long planned trip for family reunion and bonding"
        ],
        'Other': [
            "Need some days off for personal work and errands",
            "Taking break for mental health and stress relief",
            "Want to pursue personal hobby project during time off",
            "Need time to complete pending personal documentation work",
            "Planning to attend professional certification exam",
            "Personal reasons requiring few days of absence",
            "Need to handle some important personal matters",
            "Taking time off for self-care and rejuvenation",
            "Want to spend time on personal development activities",
            "Need break from work for personal well-being"
        ]
    }
    
    data = []
    
    for i in range(n_samples):
        # Generate employee data
        emp_id = f"EMP{str(i+1).zfill(5)}"
        emp_name = f"Employee_{i+1}"
        department = random.choice(departments)
        employment_status = random.choice(employment_statuses)
        
        # Generate joining date (1 day to 3 years ago)
        days_ago = random.randint(1, 1095)
        joining_date = datetime.now() - timedelta(days=days_ago)
        service_period = days_ago
        
        # Generate leave details
        leave_type = random.choice(leave_types)
        leave_duration = random.randint(1, 20)  # Including some invalid durations
        
        # Generate dates
        start_date = datetime.now() + timedelta(days=random.randint(1, 30))
        end_date = start_date + timedelta(days=leave_duration - 1)
        
        # Generate reason based on leave type with some randomness
        if leave_type == 'Sick':
            category = 'Medical' if random.random() > 0.2 else random.choice(['Personal', 'Other'])
        elif leave_type == 'Emergency':
            category = 'Emergency' if random.random() > 0.15 else random.choice(['Medical', 'Personal'])
        elif leave_type == 'Casual':
            category = random.choice(['Personal', 'Other', 'Medical'])
        else:  # Earned
            category = random.choice(['Personal', 'Medical', 'Other'])
        
        reason = random.choice(reason_templates[category])
        
        # Determine approval based on realistic logic
        approve = 1  # Default approve
        
        # Apply business rules (similar to rules.py)
        if service_period < 30:
            approve = 0
        elif leave_duration > 15:
            approve = 0
        elif employment_status == 'Contract' and leave_duration > 5:
            approve = 0
        elif employment_status == 'Probation' and leave_duration > 2:
            approve = 0
        elif leave_type == 'Casual' and leave_duration > 3:
            approve = 0
        elif leave_type == 'Earned' and service_period < 180:
            approve = 0
        elif leave_type == 'Emergency' and leave_duration <= 2:
            approve = 1  # Auto-approve
        else:
            # Add some variability for ML to learn patterns
            # Consider multiple factors
            score = 0
            
            # Positive factors
            if employment_status == 'Permanent':
                score += 3
            elif employment_status == 'Probation':
                score += 1
            
            if service_period > 365:
                score += 2
            elif service_period > 180:
                score += 1
            
            if leave_duration <= 3:
                score += 3
            elif leave_duration <= 7:
                score += 1
            
            if category == 'Medical':
                score += 2
            elif category == 'Emergency':
                score += 3
            
            if leave_type == 'Sick' and category == 'Medical':
                score += 2
            
            # Negative factors
            if leave_duration > 10:
                score -= 2
            
            if category == 'Other' and leave_duration > 5:
                score -= 1
            
            # Decision based on score
            if score >= 5:
                approve = 1
            elif score <= 2:
                approve = 0
            else:
                approve = 1 if random.random() > 0.3 else 0
        
        data.append({
            'employee_id': emp_id,
            'employee_name': emp_name,
            'department': department,
            'employment_status': employment_status,
            'joining_date': joining_date.strftime('%Y-%m-%d'),
            'service_period': service_period,
            'leave_type': leave_type,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'leave_duration': leave_duration,
            'reason': reason,
            'reason_category': category,
            'approved': approve
        })
    
    return pd.DataFrame(data)

def prepare_features(df, tfidf_vectorizer):
    """
    Prepare feature matrix for ML model.
    """
    # Encode categorical variables
    employment_encoding = {'Permanent': 2, 'Probation': 1, 'Contract': 0}
    leave_type_encoding = {'Sick': 0, 'Casual': 1, 'Earned': 2, 'Emergency': 3}
    category_encoding = {'Medical': 0, 'Personal': 1, 'Emergency': 2, 'Other': 3}
    
    df['employment_encoded'] = df['employment_status'].map(employment_encoding)
    df['leave_type_encoded'] = df['leave_type'].map(leave_type_encoding)
    df['category_encoded'] = df['reason_category'].map(category_encoding)
    
    # Get TF-IDF vectors for reasons
    tfidf_vectors = []
    for reason in df['reason']:
        vector = get_reason_embedding(reason, tfidf_vectorizer)
        tfidf_vectors.append(vector)
    
    tfidf_array = np.array(tfidf_vectors)
    
    # Combine all features
    basic_features = df[['service_period', 'leave_duration', 'employment_encoded', 
                         'leave_type_encoded', 'category_encoded']].values
    
    X = np.concatenate([basic_features, tfidf_array], axis=1)
    y = df['approved'].values
    
    return X, y

def train_model(X_train, y_train, model_type='logistic'):
    """
    Train ML model.
    """
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:  # decision_tree
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
    
    model.fit(X_train, y_train)
    return model

def main():
    print("=" * 60)
    print("HR LEAVE ANALYZER - ML MODEL TRAINING")
    print("=" * 60)
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Generate synthetic dataset
    print("\n[1] Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=1000)
    
    # Save dataset
    dataset_path = 'data/leave_dataset.csv'
    df.to_csv(dataset_path, index=False)
    print(f"    Dataset saved to {dataset_path}")
    print(f"    Total samples: {len(df)}")
    print(f"    Approved: {df['approved'].sum()} ({df['approved'].sum()/len(df)*100:.1f}%)")
    print(f"    Rejected: {len(df) - df['approved'].sum()} ({(len(df) - df['approved'].sum())/len(df)*100:.1f}%)")
    
    # Step 2: Create TF-IDF vectorizer
    print("\n[2] Creating TF-IDF vectorizer for NLP features...")
    tfidf_vectorizer = create_tfidf_vectorizer(df['reason'].tolist(), max_features=50)
    
    # Save vectorizer
    vectorizer_path = 'models/tfidf_vectorizer.joblib'
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"    Vectorizer saved to {vectorizer_path}")
    
    # Step 3: Prepare features
    print("\n[3] Preparing feature matrix...")
    X, y = prepare_features(df, tfidf_vectorizer)
    print(f"    Feature matrix shape: {X.shape}")
    
    # Step 4: Split data
    print("\n[4] Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples: {len(X_test)}")
    
    # Step 5: Train model
    print("\n[5] Training Logistic Regression model...")
    model = train_model(X_train, y_train, model_type='logistic')
    
    # Save model
    model_path = 'models/ml_model.joblib'
    joblib.dump(model, model_path)
    print(f"    Model saved to {model_path}")
    
    # Step 6: Evaluate model
    print("\n[6] Evaluating model performance...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\n    Training Accuracy: {train_accuracy:.4f}")
    print(f"    Testing Accuracy: {test_accuracy:.4f}")
    
    print("\n    Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, 
                                target_names=['Rejected', 'Approved']))
    
    print("\n    Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nYou can now run the Flask application:")
    print("    python app.py")
    print("\nThen open your browser and go to:")
    print("    http://127.0.0.1:5000")
    print("=" * 60)

if __name__ == '__main__':
    main()