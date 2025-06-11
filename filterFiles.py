import pandas as pd
import numpy as np

def filter_csv_by_un_members():
    """
    Filter income and educational CSV files to include only UN recognized countries
    """
    
    try:
        # Read the UN members file
        print("Reading UN members file...")
        un_members = pd.read_csv('member_state_auths_2025-03-14.csv')  # Replace with your actual filename
        
        # Get the list of UN member ISO codes
        un_iso_codes = set(un_members['ISO Code'].dropna().str.strip())
        print(f"Found {len(un_iso_codes)} UN member countries")
        
        # Read the income CSV file
        print("Reading income CSV file...")
        income_df = pd.read_csv('income.csv', sep=';')  # Note: using semicolon separator
        
        # Clean the Code column in income file
        income_df['Code'] = income_df['Code'].str.strip()
        
        # Filter income data to include only UN members
        income_filtered = income_df[income_df['Code'].isin(un_iso_codes)]
        print(f"Income file: {len(income_df)} total rows, {len(income_filtered)} UN member rows")
        
        # Read the educational CSV file
        print("Reading educational CSV file...")
        # Try different encodings to handle special characters
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        education_df = None
        for encoding in encodings_to_try:
            try:
                education_df = pd.read_csv('Educational attainment by level of education, cumulative (% population 25+).csv', encoding=encoding)
                print(f"Successfully read education file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if education_df is None:
            raise Exception("Could not read education.csv with any common encoding")
        
        # Clean the Economy Code column in education file
        education_df['Economy Code'] = education_df['Economy Code'].str.strip()
        
        # Filter education data to include only UN members
        education_filtered = education_df[education_df['Economy Code'].isin(un_iso_codes)]
        print(f"Education file: {len(education_df)} total rows, {len(education_filtered)} UN member rows")
        
        # Save the filtered files
        income_filtered.to_csv('income_un_members_only.csv', sep=';', index=False)
        education_filtered.to_csv('education_un_members_only.csv', index=False)
        
        print("\nFiltered files saved:")
        print("- income_un_members_only.csv")
        print("- education_un_members_only.csv")
        
        # Display some statistics
        print(f"\nSummary:")
        print(f"UN member countries: {len(un_iso_codes)}")
        print(f"Income data for UN members: {len(income_filtered)} countries")
        print(f"Education data for UN members: {len(education_filtered)} countries")
        
        # Show which countries from the sample data are UN members
        sample_countries = ['AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'ATG', 'ARG', 'ARM']
        un_sample = [code for code in sample_countries if code in un_iso_codes]
        non_un_sample = [code for code in sample_countries if code not in un_iso_codes]
        
        print(f"\nFrom your sample data:")
        print(f"UN members: {un_sample}")
        print(f"Non-UN members: {non_un_sample}")
        
        # Preview the filtered data
        print(f"\nPreview of filtered income data:")
        print(income_filtered.head())
        
        print(f"\nPreview of filtered education data:")
        print(education_filtered.head())
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please make sure all CSV files are in the same directory as this script")
        print("Expected files: income.csv, education.csv, un_members.csv")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    filter_csv_by_un_members()