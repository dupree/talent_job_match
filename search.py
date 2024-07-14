import joblib
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from train import extract_features

class Search:
    def __init__(self, model_path='logistic_regression_model.pkl') -> None:
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Loads a pre-trained model from the given path using joblib.

        Args:
        - model_path (str): Path to the pre-trained model file.

        Returns:
        - model: Loaded model object.
        """
        try:
            model = joblib.load(model_path)
            return model
        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' not found.")
            raise

    def predict_match(self, talent, job):
        """
        Predicts the match and score between a talent and a job using the loaded model.

        Args:
        - talent (dict): Dictionary representing the talent's attributes.
        - job (dict): Dictionary representing the job's attributes.

        Returns:
        - tuple: A tuple containing the predicted match (bool) and score (float).
        """
        features = extract_features(talent, job)
        match = self.model.predict([features])[0]
        score = self.model.predict_proba([features])[0][1]  # Probability of class 1 (match)
        return match, score

    def match(self, talent: dict, job: dict) -> dict:
        """
        Matches a talent and a job using the predict_match method.

        Args:
        - talent (dict): Dictionary representing the talent's attributes.
        - job (dict): Dictionary representing the job's attributes.

        Returns:
        - dict: A dictionary containing 'talent', 'job', 'label' (match result), and 'score'.
        """
        match, score = self.predict_match(talent, job)
        return {
            'talent': talent,
            'job': job,
            'label': match,
            'score': score
        }

    def match_bulk(self, talents: list[dict], jobs: list[dict]) -> list[dict]:
        """
        Matches multiple talents with multiple jobs and ranks results by score.

        Args:
        - talents (list of dict): List of dictionaries representing talents' attributes.
        - jobs (list of dict): List of dictionaries representing jobs' attributes.

        Returns:
        - list of dict: A list of dictionaries, each containing 'talent', 'job', 'label' (match result), and 'score'.
        """
        results = []
        for talent in talents:
            for job in jobs:
                result = self.match(talent, job)
                result['label'] = bool(result['label'])
                results.append(result)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python search.py <model_path> <talents_path> <jobs_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    talents_path = sys.argv[2]
    jobs_path = sys.argv[3]
    output_path = "search_results.txt"

    try:
        search = Search(model_path)

        with open(talents_path, 'r') as f:
            talents = json.load(f)

        with open(jobs_path, 'r') as f:
            jobs = json.load(f)

        ranked_results = search.match_bulk(talents, jobs)

        with open(output_path, 'w') as f:
            for result in ranked_results:
                f.write(json.dumps(result) + "\n")

        print(f"Results written to {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
