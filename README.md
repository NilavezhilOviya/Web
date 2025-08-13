# Credit Card Fraud Detection

This project detects fraudulent transactions using machine learning models on the **fraudTrain.csv** and **fraudTest.csv** datasets.  
The dataset contains anonymized features related to user transactions, along with a label indicating whether a transaction is fraudulent.

## 📂 Dataset
- **Source**: Synthetic credit card fraud detection dataset
- Files used:
  - `fraudTrain.csv`
  - `fraudTest.csv`
- Label column: `is_fraud` (1 = Fraud, 0 = Legitimate)

## 🚀 How to Run on Google Colab

1. Open the Colab notebook:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_NOTEBOOK_LINK)

2. Upload the dataset:
   - Click on the folder icon in the left sidebar in Colab.
   - Upload `fraudTrain.csv` and `fraudTest.csv`.

3. Install dependencies (only once per session):
   ```python
   !pip install pandas numpy scikit-learn matplotlib seaborn
4. Import libraries:
   ```python
    X_train = train_df.drop(columns=['is_fraud'])
    y_train = train_df['is_fraud']

    X_test = test_df.drop(columns=['is_fraud'])
    y_test = test_df['is_fraud']

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

5. Load the data:
   ```python
   train_df = pd.read_csv("fraudTrain.csv")
   test_df = pd.read_csv("fraudTest.csv")
6. Train the model:
   ```python
       X_train = train_df.drop(columns=['is_fraud'])
       y_train = train_df['is_fraud']
    
       X_test = test_df.drop(columns=['is_fraud'])
       y_test = test_df['is_fraud']
    
       model = RandomForestClassifier(random_state=42)
       model.fit(X_train, y_train)
7. Evaluate the model:
```python
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
    plt.show()


