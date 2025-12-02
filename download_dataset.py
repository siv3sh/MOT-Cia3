"""
Helper script to download the Breast Cancer Wisconsin dataset.
This script provides multiple options for downloading the dataset.
"""

import os
import urllib.request
import zipfile


def download_from_url():
    """
    Attempt to download dataset directly (may not work due to Kaggle authentication).
    """
    url = "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/download"
    output_path = "data/data.csv"
    
    print("Attempting direct download...")
    try:
        os.makedirs('data', exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Dataset downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Direct download failed: {e}")
        return False


def instructions():
    """
    Print instructions for manual download.
    """
    print("\n" + "="*70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nOption 1: Manual Download (Recommended)")
    print("  1. Visit: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data")
    print("  2. Click 'Download' button")
    print("  3. Extract the ZIP file")
    print("  4. Copy 'data.csv' to the 'data/' folder in this project")
    print("  5. Ensure the file is named 'data.csv'")
    
    print("\nOption 2: Using Kaggle API")
    print("  If you have Kaggle API credentials set up:")
    print("  kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p data/")
    print("  cd data && unzip breast-cancer-wisconsin-data.zip")
    print("  mv *.csv data.csv")
    
    print("\nOption 3: Alternative Source")
    print("  The dataset is also available on UCI ML Repository:")
    print("  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)")
    print("="*70 + "\n")


if __name__ == "__main__":
    if not download_from_url():
        instructions()


