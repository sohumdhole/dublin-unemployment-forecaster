import pandas as pd
import numpy as np

def prepare_data(input_file="dataset.csv", output_file="cleaned_unemployment.csv"):
    try:
        # The first 9 rows contain messy multi-line headers. 
        # Row 10 (index 9 in zero-indexed) is the start of actual data: Q1 98,9.0%,43.4,...
        
        # We will read exactly the columns we care about, skipping the messy headers.
        df = pd.read_csv(input_file, skiprows=9, header=None)
        
        # Columns based on our analysis:
        # 0: Quarter
        # 1: National Unemployment Rate SA (%)
        # 2: Dublin Unemployed SA (000)
        # 3: Dublin Unemployment Rate SA (%)
        # 4: Dublin Employed SA (000)
        
        # Assign clear names
        df.columns = [
            "Quarter", "National_Unemployment_Rate_Pct", 
            "Dublin_Unemployed_Thousands", "Dublin_Unemployment_Rate_Pct", 
            "Dublin_Employed_Thousands"
        ] + [f"Drop_{i}" for i in range(5, len(df.columns))]
        
        # Keep only the relevant columns
        df = df[[
            "Quarter", "National_Unemployment_Rate_Pct", 
            "Dublin_Unemployed_Thousands", "Dublin_Unemployment_Rate_Pct", 
            "Dublin_Employed_Thousands"
        ]]
        
        # Drop rows where Quarter is NaN or doesn't start with 'Q'
        df = df.dropna(subset=['Quarter'])
        df = df[df['Quarter'].str.startswith('Q')]
        
        # Clean percentage strings (e.g., "9.0%" -> 9.0)
        for col in ["National_Unemployment_Rate_Pct", "Dublin_Unemployment_Rate_Pct"]:
            df[col] = df[col].astype(str).str.replace("%", "").astype(float)
            
        # Ensure numerical columns are floats, handle errors with NaN
        numeric_cols = ["Dublin_Unemployed_Thousands", "Dublin_Employed_Thousands"]
        for col in numeric_cols:
            # Replace empty strings or whitespace with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert Quarter (e.g., "Q1 98") to Datetime
        def parse_quarter(q_str):
            # q_str like "Q1 98" or "Q1 05"
            parts = q_str.strip().split()
            if len(parts) == 2:
                q, yr = parts
                # Handle Y2K window (let's say < 50 is 2000s, >= 50 is 1900s)
                yr_int = int(yr)
                if yr_int >= 50:
                    century = 1900
                else:
                    century = 2000
                full_year = century + yr_int
                
                # Map quarters to start months
                month_map = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
                month = month_map.get(q, 1)
                return pd.Timestamp(year=full_year, month=month, day=1)
            return pd.NaT

        df['Date'] = df['Quarter'].apply(parse_quarter)
        
        # Sort by Date just in case
        df = df.sort_values(by='Date')
        
        # Forward fill any random missing numeric values (if applicable)
        df.fillna(method='ffill', inplace=True)
        
        # Save to output file
        df.to_csv(output_file, index=False)
        print(f"Successfully cleaned data. Rows: {len(df)}. Saved to {output_file}.")
        return df

    except Exception as e:
        print(f"Error occurred during data preparation: {e}")

if __name__ == "__main__":
    prepare_data()
