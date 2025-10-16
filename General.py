import pandas as pd
import os

FOLDER_PATH = '.'
OUTPUT_FILENAME = 'Merged_True_False.csv'

def merge_csv_files(folder_path, output_filename):
    try:
        all_files = os.listdir(folder_path)
        csv_files = [f for f in all_files if f.endswith('.csv')]
        if len(csv_files) < 2:
            print(f"Error: Found fewer than two CSV files in the folder '{folder_path}'.")
            print("Please make sure your two CSV files are in the same folder as this script.")
            return
        if output_filename in csv_files:
            csv_files.remove(output_filename)
            print(f"Note: '{output_filename}' already exists and will be excluded from the merge.")

        print(f"Found the following CSV files to merge: {', '.join(csv_files)}")
        dataframe_list = []
        for filename in csv_files:
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                dataframe_list.append(df)
                print(f"Successfully read '{filename}' ({len(df)} rows).")
            except Exception as e:
                print(f"Could not read file '{filename}'. Error: {e}")
        
        if not dataframe_list:
            print("No files were successfully read. Halting process.")
            return

        print("\nMerging files...")
        merged_df = pd.concat(dataframe_list, ignore_index=True)

        output_path = os.path.join(folder_path, output_filename)

        merged_df.to_csv(output_path, index=False)

        print("-" * 30)
        print(f"✅ Success! Merged {len(dataframe_list)} files.")
        print(f"Total rows in new file: {len(merged_df)}")
        print(f"Merged file saved as: '{output_path}'")
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: The directory '{folder_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    merge_csv_files(FOLDER_PATH, OUTPUT_FILENAME)
