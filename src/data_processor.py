import pandas as pd

def load_and_process_data(file) -> pd.DataFrame:
    try:
        df = pd.read_excel(file)
        # Additional processing can be added here if needed
        return df
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")