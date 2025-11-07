#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:27:53 2025

@author: nca
"""

import pandas as pd

# Load the CSV file
file_path = "/home/nca/Downloads/cape_v_severity/checkpoint_with_Clarity_model/on_test_set_with_conformer4999.csv"  # Change to your actual file path
# Read with correct delimiter (comma)
df = pd.read_csv(file_path, delimiter=",")  

# Print first few rows & column names to verify structure
print(df.head())
print(df.columns)  # Check actual column names

column_c_name = df.columns[3]  # Get the actual name of Column C

# Remove brackets
df[column_c_name] = df[column_c_name].astype(str).str.replace(r"[\[\]]", "", regex=True)

# Save the cleaned file
output_path = "/home/nca/Downloads/cape_v_severity/checkpoint_with_Clarity_model/on_test_set_with_conformer4999.csv"
df.to_csv(output_path, index=False, sep=",")  # Use correct separator

print(f"Brackets removed from Column C. Cleaned file saved as {output_path}")

