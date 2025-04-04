import os
import json
import torch
import numpy as np
import soundfile as sf
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf


def create_manifest(audio_files):
    """Create a manifest for multiple audio files"""
    manifest = []
    for audio_file in audio_files:
        manifest.append(
            {
                "audio_filepath": os.path.abspath(audio_file),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": 2,
                "rttm_filepath": None,
                "uem_filepath": None,
            }
        )
    return manifest


def write_manifest(manifest, filepath):
    """Write manifest in JSONL format (one JSON object per line)"""
    with open(filepath, "w") as f:
        for entry in manifest:
            json.dump(entry, f)
            f.write("\n")


def parse_rttm(rttm_file):
    """Parse RTTM file and return list of segments"""
    segments = []
    with open(rttm_file, "r") as f:
        for line in f:
            # RTTM format: Type File Channel Start Duration NA NA Speaker NA NA
            parts = line.strip().split()
            # EXAMPLE: SPEAKER converted_20241202-081802-1733116682.271198 1   6.140   0.270 <NA> <NA> speaker_1 <NA> <NA>
            if len(parts) >= 8:  # Ensure we have enough parts
                segment = {
                    "start": float(parts[3]),
                    "duration": float(parts[4]),
                    "speaker": parts[7],
                }
                segments.append(segment)
    return segments


def convert_wav_files(input_files, target_sr=16000):
    """Convert WAV files to 16kHz uncompressed PCM format if not already converted"""
    converted_files = []
    os.makedirs("converted_wavs", exist_ok=True)
    
    for input_file in input_files:
        try:
            # Check if converted file already exists
            output_file = os.path.join(
                "converted_wavs",
                f"converted_{os.path.basename(input_file)}"
            )
            
            if os.path.exists(output_file):
                print(f"Using existing converted file for {input_file}")
                converted_files.append(output_file)
                continue
                
            # Read the compressed WAV file
            audio_data, original_sr = sf.read(input_file)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample if necessary
            if original_sr != target_sr:
                # Calculate new length after resampling
                new_length = int(len(audio_data) * target_sr / original_sr)
                audio_resampled = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            else:
                audio_resampled = audio_data
            
            # Save as uncompressed WAV
            sf.write(
                output_file,
                audio_resampled,
                target_sr,
                subtype='PCM_16',  # Uncompressed PCM format
                format='WAV'
            )
            converted_files.append(output_file)
            print(f"Converted {input_file} to {output_file}")
            
        except Exception as e:
            print(f"Error converting {input_file}: {str(e)}")
            continue
    
    return converted_files


def generate_speaker_segments(audio_file, rttm_file, output_dir="speaker_segments"):
    """Generate separate audio files for each speaker from RTTM file"""
    try:
        # Create output directories if they don't exist
        os.makedirs(os.path.join(output_dir, "customer_segments"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "operator_segments"), exist_ok=True)

        # Read the audio file
        audio_data, sample_rate = sf.read(audio_file)
        
        # Parse RTTM file
        segments = parse_rttm(rttm_file)
        
        # Initialize lists to store segments
        customer_segments = []
        operator_segments = []
        print("Segments: ", segments)
        # Process segments
        for segment in segments:
            start_sample = int(segment["start"] * sample_rate)
            end_sample = int((segment["start"] + segment["duration"]) * sample_rate)
            
            # Ensure we don't go out of bounds
            if end_sample > len(audio_data):
                end_sample = len(audio_data)
            if start_sample >= end_sample:
                continue
            
            # Extract the segment
            segment_audio = audio_data[start_sample:end_sample]
            
            # Assign to customer (speaker_0) or operator (speaker_1)
            print("Segment: ", segment["speaker"])
            if segment["speaker"] == "speaker_0":
                customer_segments.append(segment_audio)
            else:
                operator_segments.append(segment_audio)
        
        original_file = os.path.basename(audio_file)
        if "converted_" in original_file:
            original_file = original_file.replace("converted_", "")

        # Save concatenated segments
        if customer_segments:
            customer_audio = np.concatenate(customer_segments)
            customer_output = os.path.join(output_dir, "customer_segments", f"customer_{original_file}")
            print("Customer output: ", customer_output)
            sf.write(customer_output, customer_audio, sample_rate)
        
        if operator_segments:
            operator_audio = np.concatenate(operator_segments)
            operator_output = os.path.join(output_dir, "operator_segments", f"operator_{original_file}")
            print("Operator output: ", operator_output)
            sf.write(operator_output, operator_audio, sample_rate)
        
        print(f"Successfully generated speaker segments for {original_file}")
        return True
        
    except Exception as e:
        print(f"Error generating speaker segments for {audio_file}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False


def diarize_audio(audio_files):
    """Perform diarization on audio files"""
    # Load base configuration from yaml
    base_config = OmegaConf.load('config.yaml')
    
    # Create override configuration
    override_config = OmegaConf.create({
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 0,
        "batch_size": 1,
        "diarizer": {
            "manifest_filepath": "temp_manifest.json",
            "out_dir": "./diarization_outputs",
            "oracle_vad": False,
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": True,
                    "max_num_speakers": 2
                }
            },
            "vad": {
                "parameters": {
                    "onset": 0.3,
                    "offset": 0.3,
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.1,
                    "smoothing": "median",
                    "overlap": 0.5
                }
            }
        }
    })
    
    # Merge configurations
    diarizer_config = OmegaConf.merge(base_config, override_config)
    
    try:
        # Create manifest for files
        manifest = create_manifest(audio_files)
        write_manifest(manifest, "temp_manifest.json")
        
        # Initialize and run diarizer
        diarizer = ClusteringDiarizer(cfg=diarizer_config)
        diarizer.diarize()
        return True
        
    except Exception as e:
        print(f"Error in diarization process: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
    
    finally:
        # Clean up manifest
        if os.path.exists("temp_manifest.json"):
            os.remove("temp_manifest.json")


def process_audio_files(input_dir="sesler"):
    """Main function to process audio files"""
    # Get all WAV files
    wav_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".WAV")]
    if not wav_files:
        print(f"No WAV files found in the {input_dir} directory")
        return

    print(f"Found {len(wav_files)} WAV files")
    
    # Convert WAV files if needed
    converted_files = convert_wav_files(wav_files)
    if not converted_files:
        print("No files were successfully converted")
        return

    try:
        # Create output directories
        os.makedirs("diarization_outputs/pred_rttms", exist_ok=True)
        
        # Process each file
        for conv_file in converted_files:
            base_name = os.path.basename(conv_file)
            rttm_path = os.path.join("diarization_outputs", "pred_rttms", 
                                   base_name.replace(".WAV", ".rttm"))
            
            # Check if RTTM exists
            if not os.path.exists(rttm_path):
                print(f"RTTM file not found for {base_name}, performing diarization...")
                if not diarize_audio([conv_file]):
                    print(f"Diarization failed for {base_name}")
                    continue
            else:
                print(f"Using existing RTTM file for {base_name}")
            
            # Generate speaker segments
            generate_speaker_segments(conv_file, rttm_path)
            
    finally:
        # Don't remove converted files anymore since we want to reuse them
        pass


if __name__ == "__main__":
    process_audio_files()
