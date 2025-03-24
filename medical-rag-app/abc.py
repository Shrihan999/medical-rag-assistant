#!/usr/bin/env python3
"""
This script explores the structure of the medical QA dataset to better understand its format.
"""
from datasets import load_dataset
import pandas as pd
import json
import os

def explore_dataset(config_name="all-processed"):
    # Load the dataset with a specific configuration
    print(f"Loading dataset '{config_name}' from Hugging Face...")
    dataset = load_dataset("Malikeh1375/medical-question-answering-datasets", config_name)
    
    # Print dataset structure
    print("\nDataset Structure:")
    print(dataset)
    
    # Get information about each split
    print("\nSplits Information:")
    for split_name, split in dataset.items():
        print(f"\n{split_name} split:")
        print(f"  Number of examples: {len(split)}")
        print(f"  Features: {split.features}")

        # Convert to DataFrame for analysis
        df = split.to_pandas()

        # Show sample data
        print(f"\n  Sample from {split_name} split:")
        print(df.head(3))  # Display first 3 rows

        print(f"\n  Column statistics for {split_name}:")
        for column in df.columns:
            non_null_count = df[column].count()
            null_count = df[column].isna().sum()
            print(f"    {column}: {non_null_count} non-null, {null_count} null values")
            
            # If string column, show some length statistics
            if df[column].dtype == 'object':
                lengths = df[column].astype(str).apply(len)
                print(f"      Min length: {lengths.min()}, Max length: {lengths.max()}, Avg length: {lengths.mean():.1f}")

        # Ensure the directory exists before saving the file
        os.makedirs("data/raw", exist_ok=True)
        
        # Save a sample to disk for reference
        print("\nSaving sample data to disk...")
        sample_data = df.head(10).to_dict(orient="records")  # Convert first 10 rows to a list of dictionaries
        sample_path = f"data/raw/sample_{split_name}.json"
        with open(sample_path, "w") as f:
            json.dump(sample_data, f, indent=2)
        print(f"Sample saved to {sample_path}")

    print("Exploration complete!")

if __name__ == "__main__":
    explore_dataset()
