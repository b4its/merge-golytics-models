import os
import matplotlib.pyplot as plt
import pandas as pd

def load_clear_data(filepath):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, filepath)

    df = pd.read_csv(full_path)

    df = df.drop(columns=[
        'Account Number', 'Legal Name', 'Owner First Name', 'Owner Middle Initial',
        'Owner Last Name', 'Suffix', 'Legal Entity Owner', 'Formalization Status',

    ])

    return df

def save_processed(df, output_path='datasets/processed_data.csv'):

    df.to_csv(output_path, index=False)

    print(f'Data berhasil disimpan ke {output_path}')
