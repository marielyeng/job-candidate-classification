import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import os

def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    print("Original Data Shape:", df.shape)
    
    # Drop unnecessary columns
    df = df.drop(columns=['Id', 'PreviousSalary', 'Country'])
    
    # Drop missing values
    df = df.dropna()
    print("After Dropping Missing Values:", df.shape)
    
    # Handle the HaveWorkedWith column
    df['HaveWorkedWith'] = df['HaveWorkedWith'].str.split(';')
    df = df.explode('HaveWorkedWith')
    
    # One-Hot Encoding for HaveWorkedWith column
    df_one_hot = pd.get_dummies(df, columns=['HaveWorkedWith'], prefix='HW', drop_first=True)
    
    # Identify categorical features (update to exclude HaveWorkedWith)
    categorical_features = ['Age', 'Accessibility', 'EdLevel', 'Employment',
                            'Gender', 'MentalHealth', 'MainBranch', 'YearsCode',
                            'YearsCodePro', 'ComputerSkills']
    
    # Encode categorical features using Label Encoding
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_one_hot[col] = le.fit_transform(df_one_hot[col])
        label_encoders[col] = le
        print(f"Encoded {col}")
    
    # Encode the target variable 'Employed'
    le_grade = LabelEncoder()
    df_one_hot['Employed'] = le_grade.fit_transform(df_one_hot['Employed'])
    joblib.dump(le_grade, os.path.join(model_dir, 'label_encoder_employed.pkl'))
    print("Encoded and saved label encoder for target variable 'Employed'.")
    
    # Separate features and target variable
    X = df_one_hot.drop(columns=['Employed'])
    y = df_one_hot['Employed']
    
    # Apply Linear Discriminant Analysis (LDA)
    lda = LinearDiscriminantAnalysis(n_components=1)  # Choose n_components based on your classes
    X_lda = lda.fit_transform(X, y)  # Fit and transform the data
    print(f"Reduced features shape with LDA: {X_lda.shape}")
    
    # Save the transformed data into a DataFrame
    lda_df = pd.DataFrame(X_lda, columns=[f'LDA_Component_{i+1}' for i in range(X_lda.shape[1])])
    lda_df['Employed'] = y.reset_index(drop=True)  # Add the target variable back
    
    # Save preprocessed data
    lda_df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_csv = os.path.join('data', 'candidates.csv')
    output_csv = os.path.join('data', 'preprocessed_candidates.csv')
    preprocess_data(input_csv, output_csv)
