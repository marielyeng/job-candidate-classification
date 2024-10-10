import pandas as pd
import joblib
import os

def select_features(input_path, output_path):
    try:
        # Load preprocessed data with LDA components
        df = pd.read_csv(input_path)
        print("Preprocessed Data Shape:", df.shape)
        
        # Save the names of the LDA components (features) as they are already reduced
        selected_features = df.columns.drop('Employed')
        print("LDA Components (Selected Features):", list(selected_features))
        
        # Save selected features
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'model')
        joblib.dump(list(selected_features), os.path.join(model_dir, 'selected_features.pkl'))
        print("Saved selected features.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    input_csv = os.path.join(project_root, 'data', 'preprocessed_candidate.csv')
    select_features(input_csv, input_csv)
