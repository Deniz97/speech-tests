import os
from collections import defaultdict

def analyze_wav_filenames():
    # Get all WAV files in the sesler directory
    wav_files = [f for f in os.listdir('sesler') if f.endswith('.WAV')]
    
    # Print total file count
    print(f"Total number of WAV files: {len(wav_files)}")
    
    # Dictionaries to store parts and their occurrences
    second_parts = defaultdict(list)
    third_parts = defaultdict(list)
    
    for filename in wav_files:
        # Remove .WAV extension
        name = filename[:-4]
        
        # Split by hyphen
        parts = name.split('-')
        
        if len(parts) >= 3:
            # For files starting with OUTXXX
            if parts[0].startswith('OUT'):
                second_part = parts[1]  # YYYYMMDD
                third_part = parts[2]   # HHMMSS
            else:
                # For files without OUT prefix
                second_part = parts[0]  # YYYYMMDD
                third_part = parts[1]   # HHMMSS
            
            second_parts[second_part].append(filename)
            third_parts[third_part].append(filename)
    
    # Print results
    print("\nAnalysis of WAV filenames in sesler folder:")
    print("\nDuplicate second parts (YYYYMMDD):")
    for part, files in second_parts.items():
        if len(files) > 1:
            print(f"\nDate: {part}")
            for file in files:
                print(f"  - {file}")
    
    print("\nDuplicate third parts (HHMMSS):")
    for part, files in third_parts.items():
        if len(files) > 1:
            print(f"\nTime: {part}")
            for file in files:
                print(f"  - {file}")

if __name__ == "__main__":
    analyze_wav_filenames()
