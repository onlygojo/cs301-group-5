def create_model(data):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import pickle  # Imported pickle to save the model
    
    # Split data into features and target variable
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit the model
    model = LogisticRegression(max_iter=10000) # Increased max_iter to ensure convergence
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # SAVE THE MODEL USING PICKLE
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("\nModel successfully saved to 'model.pkl'!")
    
    return model