import nemo
import nemo.collections.asr as nemo_asr
import torch
import librosa
import numpy as np
import os
from pathlib import Path
import random

# Load a pretrained speaker verification model (ECAPA-TDNN) from NeMo
def load_model():
    # Load the pretrained speaker verification model from NeMo
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess audio and extract embedding for a speaker
def extract_embedding(audio_data, model, sr=16000):
    # Convert the audio to a Tensor and ensure it's on the same device as the model
    audio_tensor = torch.tensor(audio_data).unsqueeze(0).to(model.device)
    
    # Get signal length
    signal_length = torch.tensor([len(audio_data)]).long().to(model.device)
    
    # Extract embedding using the correct method
    with torch.no_grad():
        embedding, _ = model.forward(input_signal=audio_tensor, input_signal_length=signal_length)
    
    return embedding

# Register a user with their voice (store their embedding)
def register_user(user_id, audio_data, embeddings_db, model):
    embedding = extract_embedding(audio_data, model)
    embeddings_db[user_id] = embedding
    print(f"User {user_id} registered successfully.")

# Verify if the input speech matches the registered user's voice
def verify_user(user_id, audio_data, embeddings_db, model):
    # Extract the embedding of the new input speech
    input_embedding = extract_embedding(audio_data, model)
    
    # Compare the input embedding with the stored embedding of the registered user
    registered_embedding = embeddings_db.get(user_id)
    if registered_embedding is None:
        print("User not registered.")
        return False
    
    # Calculate cosine similarity between the embeddings
    similarity = torch.cosine_similarity(input_embedding, registered_embedding, dim=1)
    
    # If the similarity is above a threshold (e.g., 0.7), consider the verification successful
    threshold = 0.7  # Adjusting threshold to be more reasonable for real-world audio
    if similarity.item() > threshold:
        print(f"Verification successful! Similarity score: {similarity.item():.3f}")
        return True
    else:
        print(f"Verification failed. Similarity score: {similarity.item():.3f}")
        return False

def process_audio_file(file_path, model, embeddings_db, all_wav_files):
    print(f"\nProcessing file: {file_path}")
    
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Split the audio in half
    split_point = len(audio) // 2
    first_half = audio[:split_point]
    second_half = audio[split_point:]
    
    # Use the filename without extension as the user_id
    user_id = Path(file_path).stem
    
    # Register user with first half
    print("Registering user with first half of audio...")
    register_user(user_id, first_half, embeddings_db, model)
    
    # Test 1: Verify user with their own second half (should succeed)
    print("\nTest 1: Verifying with correct voice (second half of same file)...")
    result_correct = verify_user(user_id, second_half, embeddings_db, model)
    
    # Test 2: Verify against a random different file's second half (should fail)
    # Choose a random different file
    other_files = [f for f in all_wav_files if f != os.path.basename(file_path)]
    if other_files:  # If there are other files available
        random_file = random.choice(other_files)
        # Determine the directory of the random file
        random_file_dir = "speaker_segments/customer_segments" if "customer" in random_file else "speaker_segments/operator_segments"
        random_file_path = os.path.join(random_file_dir, random_file)
        print(f"\nTest 2: Verifying with incorrect voice (second half of {random_file})...")
        
        # Load and split the random file
        random_audio, _ = librosa.load(random_file_path, sr=16000)
        random_second_half = random_audio[len(random_audio)//2:]
        
        result_incorrect = verify_user(user_id, random_second_half, embeddings_db, model)
    else:
        print("\nTest 2: Skipped - no other files available for negative testing")
        result_incorrect = None
    
    return {
        'correct_verification': result_correct,
        'incorrect_verification': result_incorrect
    }

# Example of using the system
if __name__ == "__main__":
    # Initialize model
    print("Loading model...")
    model = load_model()

    # Initialize a dictionary to store user embeddings
    embeddings_db = {}

    # Get files from both directories
    customer_dir = "speaker_segments/customer_segments"
    operator_dir = "speaker_segments/operator_segments"
    
    # Collect all WAV files from both directories with their full paths
    wav_files = []
    for directory in [customer_dir, operator_dir]:
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith('.WAV') or f.endswith('.wav')]
            wav_files.extend([(f, os.path.join(directory, f)) for f in files])
    
    print(f"\nFound {len(wav_files)} WAV files to process:")
    for filename, filepath in wav_files:
        print(f"- {filepath}")
    
    results = {}
    # Pass the list of all filenames for random selection
    all_filenames = [f[0] for f in wav_files]
    
    for filename, filepath in wav_files:
        results[filename] = process_audio_file(filepath, model, embeddings_db, all_filenames)
    
    # Print summary
    print("\nProcessing Summary:")
    print("-----------------")
    for file_name, result in results.items():
        source_type = "Customer" if "customer_segments" in file_name else "Operator"
        print(f"\n{source_type} file - {file_name}:")
        print(f"  - Correct voice verification: {'✓ Passed' if result['correct_verification'] else '✗ Failed'}")
        if result['incorrect_verification'] is not None:
            print(f"  - Incorrect voice verification: {'✗ Failed (Good!)' if not result['incorrect_verification'] else '✓ Passed (Bad!)'}")
        else:
            print("  - Incorrect voice verification: Skipped")
