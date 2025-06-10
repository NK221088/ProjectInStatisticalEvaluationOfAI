import pandas as pd
import csv

def filter_gdp_by_un_membership(member_states_file, gdp_file, output_file):
    """
    Filter GDP data to only include countries that are UN member states.
    
    Args:
        member_states_file (str): Path to the UN member states CSV file
        gdp_file (str): Path to the GDP CSV file  
        output_file (str): Path for the filtered output CSV file
    """
    
    # Read the UN member states file and extract ISO codes
    print("Reading UN member states file...")
    try:
        # The file appears to be comma-separated but pandas is reading it as one column
        # Let's force comma separation
        member_states_df = pd.read_csv(member_states_file, sep=',')
        
        # If we still get one column, the file might have a different encoding or format
        if len(member_states_df.columns) == 1:
            # Try with different encoding
            try:
                member_states_df = pd.read_csv(member_states_file, sep=',', encoding='utf-8-sig')
            except:
                try:
                    member_states_df = pd.read_csv(member_states_file, sep=',', encoding='latin-1')
                except:
                    # Last resort - try tab separation
                    member_states_df = pd.read_csv(member_states_file, sep='\t')
        
        print(f"Available columns in member states file: {list(member_states_df.columns)}")
        
        # Look for the ISO Code column (handle potential variations)
        iso_column = None
        for col in member_states_df.columns:
            if 'iso' in col.lower() and 'code' in col.lower():
                iso_column = col
                break
        
        if iso_column is None:
            print("Error: Could not find ISO Code column in member states file")
            print("Trying to use column position 2 (assuming ISO Code is 3rd column)...")
            if len(member_states_df.columns) >= 3:
                iso_column = member_states_df.columns[2]
            else:
                return False
            
        print(f"Using column '{iso_column}' for ISO codes")
        
        # Extract ISO codes and convert to set for faster lookup
        un_iso_codes = set(member_states_df[iso_column].dropna().astype(str).str.strip())
        print(f"Found {len(un_iso_codes)} UN member state ISO codes")
        
    except Exception as e:
        print(f"Error reading member states file: {e}")
        return False
    
    # Read the GDP file
    print("Reading GDP file...")
    try:
        # Read GDP file with error handling for inconsistent fields
        gdp_df = pd.read_csv(gdp_file, on_bad_lines='skip')
        
        # Clean up column names (remove extra spaces)
        gdp_df.columns = gdp_df.columns.str.strip()
        
        print(f"Original GDP data has {len(gdp_df)} entries")
        
    except Exception as e:
        print(f"Error reading GDP file: {e}")
        # Try alternative reading methods
        try:
            print("Trying alternative reading method...")
            gdp_df = pd.read_csv(gdp_file, sep=',', quoting=1, on_bad_lines='skip')
            gdp_df.columns = gdp_df.columns.str.strip()
            print(f"Successfully read GDP data with {len(gdp_df)} entries")
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return False
    
    # Filter GDP data to only include UN member states
    print("Filtering GDP data...")
    
    # Look for the Country Code column specifically
    print(f"Available columns in GDP file: {list(gdp_df.columns)}")
    
    code_column = None
    # First try to find "Country Code" specifically
    for col in gdp_df.columns:
        if col.strip().lower() == 'country code':
            code_column = col
            break
    
    # If not found, look for other variations
    if code_column is None:
        for col in gdp_df.columns:
            if 'code' in col.lower() or 'iso' in col.lower():
                code_column = col
                break
    
    if code_column is None:
        print("Error: Could not find country code column in GDP file")
        print(f"Available columns: {list(gdp_df.columns)}")
        return False
    
    print(f"Using column '{code_column}' for country codes")
    
    # Clean up the country codes in GDP data
    gdp_df[code_column] = gdp_df[code_column].astype(str).str.strip()
    
    # Debug: Show some examples of what we're working with
    print(f"\nFirst 10 rows of GDP data:")
    for i, row in gdp_df.head(10).iterrows():
        country_name = row.get('Country Name', 'N/A')
        country_code = row.get(code_column, 'N/A')
        print(f"  {country_name} | {country_code} | Type: {type(country_code)}")
    
    print(f"\nFirst 10 UN ISO codes: {list(un_iso_codes)[:10]}")
    
    # Check for matches with detailed debugging
    gdp_codes = set()
    print(f"\nProcessing GDP country codes:")
    for i, code in enumerate(gdp_df[code_column]):
        clean_code = str(code).strip() if pd.notna(code) else 'NaN'
        gdp_codes.add(clean_code)
        if i < 5:  # Show first 5 for debugging
            print(f"  Row {i}: '{code}' -> '{clean_code}' (in UN list: {clean_code in un_iso_codes})")
    
    # Remove 'NaN' and 'nan' from the set
    gdp_codes.discard('NaN')
    gdp_codes.discard('nan')
    gdp_codes.discard('None')
    
    matches = gdp_codes.intersection(un_iso_codes)
    print(f"\nCountries in both datasets: {len(matches)}")
    print(f"Some matching codes: {sorted(list(matches))[:15]}")
    
    # Show countries in GDP but not in UN list
    gdp_only = gdp_codes - un_iso_codes
    print(f"\nCountries in GDP but not UN members: {len(gdp_only)}")
    if gdp_only:
        print(f"Examples: {sorted(list(gdp_only))[:10]}")
    
    # Show UN countries not in GDP
    un_only = un_iso_codes - gdp_codes
    print(f"\nUN members not in GDP data: {len(un_only)}")
    if un_only:
        print(f"Examples: {sorted(list(un_only))[:10]}")
    
    # Filter: ONLY remove entries where ISO code is NOT in UN member states
    # Keep entries even if they have missing GDP data, as long as the ISO code matches
    def should_keep_row(code):
        if pd.isna(code):
            return False  # Remove only if no country code at all
        clean_code = str(code).strip()
        return clean_code in un_iso_codes
    
    mask = gdp_df[code_column].apply(should_keep_row)
    filtered_gdp_df = gdp_df[mask].copy()
    
    print(f"\nFiltered GDP data has {len(filtered_gdp_df)} entries")
    print(f"Removed {len(gdp_df) - len(filtered_gdp_df)} entries")
    
    # Show what got filtered out (should only be non-UN countries)
    removed_df = gdp_df[~mask]
    print(f"\nExamples of what was removed (should only be non-UN countries):")
    for i, row in removed_df.head(10).iterrows():
        country_name = row.get('Country Name', 'N/A')
        country_code = row.get(code_column, 'N/A')
        if pd.isna(country_code):
            reason = "No country code"
        else:
            clean_code = str(country_code).strip()
            reason = f"'{clean_code}' not a UN member state"
        print(f"  - {country_name} ({country_code}) - {reason}")
    
    # Verify that Afghanistan and other UN members are kept
    afghanistan_check = gdp_df[gdp_df[code_column].astype(str).str.strip() == 'AFG']
    if len(afghanistan_check) > 0:
        print(f"\n✅ Afghanistan (AFG) status:")
        for _, row in afghanistan_check.iterrows():
            kept = row.name in filtered_gdp_df.index
            print(f"  Country: {row.get('Country Name', 'N/A')}, Code: {row.get(code_column, 'N/A')}, Kept: {kept}")
    else:
        print(f"\n❌ Afghanistan (AFG) not found in GDP data")
    
    # Save the filtered data
    print(f"Saving filtered data to {output_file}...")
    filtered_gdp_df.to_csv(output_file, index=False)
    
    # Print some statistics
    print("\n=== FILTERING SUMMARY ===")
    print(f"Original GDP entries: {len(gdp_df)}")
    print(f"UN member states: {len(un_iso_codes)}")
    print(f"Filtered GDP entries: {len(filtered_gdp_df)}")
    print(f"Entries removed: {len(gdp_df) - len(filtered_gdp_df)}")
    
    # Show some examples of removed entries
    removed_entries = gdp_df[~gdp_df[code_column].isin(un_iso_codes)]
    if len(removed_entries) > 0:
        print(f"\nExamples of removed entries:")
        for i, row in removed_entries.head(5).iterrows():
            country_name = row.get('Country Name', 'Unknown')
            country_code = row.get(code_column, 'Unknown')
            print(f"  - {country_name} ({country_code})")
    
    return True

def main():
    """Main function to run the filtering process"""
    
    # File paths - adjust these to match your actual file names
    member_states_file = "member_state_auths_2025-03-14.csv"
    gdp_file = "GDP2023.csv"  # Assuming .csv extension
    output_file = "GDP2023_UN_members_only.csv"
    
    print("=== GDP Data Filter for UN Member States ===")
    print(f"Member states file: {member_states_file}")
    print(f"GDP file: {gdp_file}")
    print(f"Output file: {output_file}")
    print()
    
    success = filter_gdp_by_un_membership(member_states_file, gdp_file, output_file)
    
    if success:
        print(f"\n✅ Filtering completed successfully!")
        print(f"Filtered data saved to: {output_file}")
    else:
        print(f"\n❌ Filtering failed. Please check the error messages above.")

if __name__ == "__main__":
    main()