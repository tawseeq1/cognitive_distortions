import pandas as pd
from .targetwords import * 

# Map variable names to string names for the report
DISTORTION_MAP = {
    'target_catastrophizing': 'Catastrophizing',
    'target_dichotomous_Reasoning': 'Dichotomous Reasoning',
    'target_disqualifying_the_positives': 'Disqualifying the Positives',
    'target_emotional_reasoning': 'Emotional Reasoning',
    'target_fortune_telling': 'Fortune Telling',
    'target_labeling_and_mislabeling': 'Labeling and Mislabeling',
    'target_magnification_and_minimization': 'Magnification and Minimization',
    'target_mental_filtering': 'Mental Filtering',
    'target_mindreading': 'Mindreading',
    'target_overgeneralizing': 'Overgeneralizing',
    'target_personalizing': 'Personalizing',
    'target_should_statements': 'Should Statements'
}

class DistortionDetector:
    def __init__(self):
        self.distortion_dictionaries = {}
        # Dynamically load lists from the imported module
        for var_name, nice_name in DISTORTION_MAP.items():
            if var_name in globals():
                self.distortion_dictionaries[nice_name] = globals()[var_name]
            else:
                print(f"Warning: {var_name} not found in targetwords.py")

    def detect(self, sentences_df):
        """
        Scans sentences for distortions.
        Adds boolean columns for each distortion type.
        """
        print("Detecting distortions...")
        
        # detect(...)
        print("Detecting distortions...")
        
        # Inefficient but simple iteration for now, can be optimized with regex if needed
        # Given n-grams can be phrases, simple 'in' check is safest
        
        def check_distortions(text):
            hits = {}
            text_lower = text.lower()
            for distortion, ngrams in self.distortion_dictionaries.items():
                hits[distortion] = any(ngram in text_lower for ngram in ngrams)
            return pd.Series(hits)

        distortions_found = sentences_df['sentence'].apply(check_distortions)
        
        # Merge results back
        result_df = pd.concat([sentences_df, distortions_found], axis=1)
        
        return result_df, list(self.distortion_dictionaries.keys())
