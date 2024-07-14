import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

degree_hierarchy = {'none': 0, 'apprenticeship': 1, 'bachelor': 2, 'master': 3, 'doctorate': 4 }
proficiency_hierarchy = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}

def load_data(file_path):
  """
  Loads data from a JSON file into a pandas DataFrame.

  Args:
      file_path (str): Path to the JSON file containing the data.

  Returns:
      pd.DataFrame: The loaded DataFrame containing the data.

  Raises:
      FileNotFoundError: If the specified file is not found.
  """
  try:
    return pd.read_json(file_path)
  except FileNotFoundError:
    print(f"Error: File not found - {file_path}")
    raise

def extract_features(talent, job):
    """
    Extracts features relevant to talent-job matching from a talent dictionary and a job dictionary.
    
    Args:
      talent (dict): Dictionary containing information about a talent.
      job (dict): Dictionary containing information about a job.
    
    Returns:
      list: A list of features representing the talent's suitability for the job.
    """
    # Compute seniority_match
    seniority_match = 1 if talent['seniority'] in job['seniorities'] else 0
    
    # Compute degrees_match
    degrees_match = 1 if degree_hierarchy[talent['degree'].lower()] >= degree_hierarchy[job['min_degree'].lower()] else 0
    
    # Compute salary_match
    salary_match = job['max_salary'] / talent['salary_expectation']
    
    # Compute language_match
    language_match = 0
    for job_lang in job['languages']:
        job_title = job_lang['title'].lower()
        job_rating = proficiency_hierarchy[job_lang['rating']]
        for talent_lang in talent['languages']:
            talent_title = talent_lang['title'].lower()
            talent_rating = proficiency_hierarchy[talent_lang['rating']]
            if job_title == talent_title and talent_rating >= job_rating:
                language_match = 1
                break
        if language_match == 1:
            break
    
    # Compute job_role_match
    job_role_match = 1 if set(talent['job_roles']).intersection(set(job['job_roles'])) else 0
    
    # Return the feature vector
    return [seniority_match, degrees_match, salary_match, language_match, job_role_match]

def extract_features_from_df(df):
    """
    Extracts features for all talent-job pairs in a DataFrame.
    
    Args:
      df (pd.DataFrame): DataFrame containing talent and job information.
    
    Returns:
      pd.DataFrame: The original DataFrame with additional feature columns.
    """
    features_list = []
    for index, row in df.iterrows():
        talent = row['talent']
        job = row['job']
        features = extract_features(talent, job)
        features_list.append(features)
    features_df = pd.DataFrame(features_list, columns=['seniority_match', 'degrees_match', 'salary_match', 'language_match', 'job_role_match'])
    df = pd.concat([df, features_df], axis=1)
    return df

def train_and_save_model(df):
    """
    Trains a logistic regression model to predict talent-job suitability and saves it.
    
    Args:
      df (pd.DataFrame): DataFrame containing features and labels for training.
    
    Returns:
      tuple: A tuple containing the trained model, X_test data, and y_test labels.
    """
    features = ['seniority_match', 'degrees_match', 'salary_match', 'language_match', 'job_role_match']
    X = df[features]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'logistic_regression_model.pkl')
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of the trained model on the test data.
    
    Args:
      model: The trained logistic regression model.
      X_test (pd.DataFrame): DataFrame containing test features.
      y_test (pd.Series): Series containing true labels for the test data.
    
    Prints the model's accuracy and a classification report to the console and saves the report to a text file.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    with open('model_performance.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
    
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

def main(file_path='data.json'):
    df = load_data(file_path)
    print("shape of the data loaded : ", df.shape)
    df = extract_features_from_df(df)
    model, X_test, y_test = train_and_save_model(df)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'data.json'
    main(file_path)