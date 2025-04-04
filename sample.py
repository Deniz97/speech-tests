import os
import random
import numpy as np
import soundfile as sf

def analyze_audio_file():
    # Get all WAV files in the sesler directory
    wav_files = [f for f in os.listdir('sesler') if f.endswith('.WAV')]
    
    if not wav_files:
        print("No WAV files found in the sesler directory")
        return
    
    # Randomly select one file
    selected_file = random.choice(wav_files)
    file_path = os.path.join('sesler', selected_file)
    
    print(f"Analyzing file: {selected_file}")
    
    try:
        # Read the WAV file
        audio_data, sample_rate = sf.read(file_path)
        
        # Get audio properties
        n_channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
        duration = len(audio_data) / sample_rate
        
        print("\nAudio Properties:")
        print(f"Number of channels: {n_channels}")
        print(f"Sample rate (Hz): {sample_rate}")
        print(f"Duration (seconds): {duration:.2f}")
        print(f"Number of samples: {len(audio_data)}")
        
        # Additional information
        print("\nAdditional Information:")
        print(f"File size: {os.path.getsize(file_path) / 1024:.2f} KB")
        print(f"Audio data type: {audio_data.dtype}")
        print(f"Audio data shape: {audio_data.shape}")
        
        # Basic statistics
        print("\nAudio Statistics:")
        print(f"Minimum amplitude: {np.min(audio_data):.4f}")
        print(f"Maximum amplitude: {np.max(audio_data):.4f}")
        print(f"Mean amplitude: {np.mean(audio_data):.4f}")
        print(f"Standard deviation: {np.std(audio_data):.4f}")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")

if __name__ == "__main__":
    analyze_audio_file()
