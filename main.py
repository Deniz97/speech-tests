import nemo
import nemo.collections.asr as nemo_asr
import torch
import librosa
import numpy as np
import os
from pathlib import Path
import random
import argparse
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class VerificationResult:
    verified: bool
    similarity: float
    segment_index: int
    source_segment_index: int

@dataclass
class FileTestResults:
    file_name: str
    source_type: str  # "Customer" or "Operator"
    registration_segment_index: int  # Index of the segment used for registration
    num_verification_segments: int  # Number of segments used for verification
    positive_results: List[VerificationResult]
    negative_results: List[Tuple[str, VerificationResult]]  # (source_file, result)
    error: Optional[str] = None
    file_duration_sec: float = 0.0  # Duration of the audio file in seconds
    num_segments: int = 0  # Total number of segments possible
    is_too_short: bool = False  # Flag specifically for too-short files

@dataclass
class TestSummary:
    total_files: int
    successful_files: int
    failed_files: int
    total_positive_tests: int
    successful_positive_tests: int
    total_negative_tests: int
    successful_negative_tests: int
    avg_positive_similarity: float
    avg_negative_similarity: float
    results_by_source: Dict[str, Dict[str, int]]  # source_type -> metrics

# Load a pretrained speaker verification model (ECAPA-TDNN) from NeMo
def load_model():
    # Load the pretrained speaker verification model from NeMo
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model.eval()  # Set the model to evaluation mode
    return model

def split_audio_into_segments(audio: np.ndarray, segment_length_sec: float, sr: int = 16000) -> List[np.ndarray]:
    """Split audio into segments of N seconds."""
    segment_length_samples = int(segment_length_sec * sr)
    segments = []
    
    # Calculate number of complete segments
    num_segments = len(audio) // segment_length_samples
    
    for i in range(num_segments):
        start = i * segment_length_samples
        end = start + segment_length_samples
        segment = audio[start:end]
        segments.append(segment)
    
    return segments

def get_random_segments_from_others(file_path: str, all_wav_files: List[str], 
                                  num_segments: int, segment_length_sec: float,
                                  sr: int = 16000) -> List[np.ndarray]:
    """Get random segments from other files for negative testing.
    Each segment will come from a different file."""
    other_files = [f for f in all_wav_files if f != os.path.basename(file_path)]
    segments = []
    used_files = set()
    
    # Shuffle the files to randomize selection
    random.shuffle(other_files)
    
    for random_file in other_files:
        if len(segments) >= num_segments:
            break
            
        if random_file in used_files:
            continue
            
        # Determine the directory of the random file
        random_file_dir = "speaker_segments/customer_segments" if "customer" in random_file else "speaker_segments/operator_segments"
        random_file_path = os.path.join(random_file_dir, random_file)
        
        try:
            random_audio, _ = librosa.load(random_file_path, sr=sr)
            random_segments = split_audio_into_segments(random_audio, segment_length_sec, sr)
            
            if random_segments:
                segments.append(random.choice(random_segments))
                used_files.add(random_file)
        except Exception as e:
            print(f"Warning: Could not process {random_file_path}: {str(e)}")
    
    if len(segments) < num_segments:
        print(f"Warning: Could only get {len(segments)} negative samples from different files " 
              f"(requested {num_segments})")
    
    return segments

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
def register_user(user_id: str, audio_segments: List[np.ndarray], embeddings_db: Dict, model) -> None:
    """Register user with multiple audio segments."""
    embeddings = []
    for segment in audio_segments:
        embedding = extract_embedding(segment, model)
        embeddings.append(embedding)
    
    # Store all embeddings for the user
    embeddings_db[user_id] = embeddings

def verify_user(user_id: str, audio_data: np.ndarray, embeddings_db: Dict, model,
                threshold: float = 0.7) -> Tuple[bool, float]:
    """Verify user against all registered segments."""
    if user_id not in embeddings_db:
        print("User not registered.")
        return False, 0.0
    
    # Extract embedding for the test segment
    test_embedding = extract_embedding(audio_data, model)
    
    # Compare against all stored embeddings
    registered_embeddings = embeddings_db[user_id]
    max_similarity = -1.0
    
    for reg_embedding in registered_embeddings:
        similarity = torch.cosine_similarity(test_embedding, reg_embedding, dim=1)
        max_similarity = max(max_similarity, similarity.item())
    
    verified = max_similarity > threshold
    return verified, max_similarity

def process_audio_file(file_path: str, model, embeddings_db: Dict, all_wav_files: List[str],
                      segment_length_sec: float = 5.0, num_negative_samples: int = 3) -> FileTestResults:
    """Process audio file with configurable segment length and number of negative samples."""
    source_type = "Customer" if "customer_segments" in os.path.basename(file_path) else "Operator"
    file_name = os.path.basename(file_path)
    
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=16000)
        file_duration_sec = len(audio) / sr
        
        # Split audio into N-second segments
        segments = split_audio_into_segments(audio, segment_length_sec)
        
        if len(segments) < 2:
            return FileTestResults(
                file_name=file_name,
                source_type=source_type,
                registration_segment_index=-1,
                num_verification_segments=0,
                positive_results=[],
                negative_results=[],
                error=f"File too short for segmentation (duration: {file_duration_sec:.1f}s, need at least {segment_length_sec*2:.1f}s for 2 segments)",
                file_duration_sec=file_duration_sec,
                num_segments=len(segments),
                is_too_short=True
            )
        
        # Randomly select one segment for registration
        registration_index = random.randrange(len(segments))
        registration_segment = segments[registration_index]
        
        # Use all other segments for verification
        verification_segments = [seg for i, seg in enumerate(segments) if i != registration_index]
        
        # Use the filename without extension as the user_id
        user_id = Path(file_path).stem
        
        # Register user with the selected segment
        register_user(user_id, [registration_segment], embeddings_db, model)
        
        # Test positive cases: verify each remaining segment
        positive_results = []
        for i, test_segment in enumerate(verification_segments):
            verified, similarity = verify_user(user_id, test_segment, embeddings_db, model)
            # Calculate the original segment index (accounting for the removed registration segment)
            orig_index = i if i < registration_index else i + 1
            positive_results.append(VerificationResult(
                verified=verified,
                similarity=similarity,
                segment_index=orig_index,
                source_segment_index=registration_index
            ))
        
        # Get random segments from other files for negative testing
        negative_segments = get_random_segments_from_others(
            file_path, all_wav_files, num_negative_samples, segment_length_sec
        )
        
        # Test negative cases
        negative_results = []
        for i, neg_segment in enumerate(negative_segments):
            verified, similarity = verify_user(user_id, neg_segment, embeddings_db, model)
            source_file = all_wav_files[i] if i < len(all_wav_files) else "unknown"
            negative_results.append((
                source_file,
                VerificationResult(
                    verified=verified,
                    similarity=similarity,
                    segment_index=i,
                    source_segment_index=registration_index
                )
            ))
        
        return FileTestResults(
            file_name=file_name,
            source_type=source_type,
            registration_segment_index=registration_index,
            num_verification_segments=len(verification_segments),
            positive_results=positive_results,
            negative_results=negative_results,
            file_duration_sec=file_duration_sec,
            num_segments=len(segments),
            is_too_short=False
        )
        
    except Exception as e:
        try:
            # Try to get file duration even if processing failed
            audio, sr = librosa.load(file_path, sr=16000)
            file_duration_sec = len(audio) / sr
            # Check if the file is too short
            is_too_short = file_duration_sec < (segment_length_sec * 2)
        except:
            file_duration_sec = 0.0
            is_too_short = False
            
        return FileTestResults(
            file_name=file_name,
            source_type=source_type,
            registration_segment_index=-1,
            num_verification_segments=0,
            positive_results=[],
            negative_results=[],
            error=str(e),
            file_duration_sec=file_duration_sec,
            num_segments=0,
            is_too_short=is_too_short
        )

def calculate_summary(results: Dict[str, FileTestResults]) -> TestSummary:
    """Calculate summary statistics from all test results."""
    # Count files by category
    successful_files = sum(1 for r in results.values() if r.error is None)
    too_short_files = sum(1 for r in results.values() if r.is_too_short)
    other_failed_files = sum(1 for r in results.values() if r.error is not None and not r.is_too_short)
    
    # Only include successful files in test statistics
    valid_results = [r for r in results.values() if r.error is None]
    
    # Calculate positive test statistics
    total_positive_tests = 0
    successful_positive_tests = 0
    all_positive_similarities = []
    
    # Calculate negative test statistics
    total_negative_tests = 0
    successful_negative_tests = 0
    all_negative_similarities = []
    
    for result in valid_results:
        # Positive tests (should verify successfully)
        total_positive_tests += len(result.positive_results)
        successful_positive_tests += sum(1 for pr in result.positive_results if pr.verified)
        all_positive_similarities.extend(pr.similarity for pr in result.positive_results)
        
        # Negative tests (should NOT verify successfully)
        total_negative_tests += len(result.negative_results)
        successful_negative_tests += sum(1 for _, nr in result.negative_results if not nr.verified)
        all_negative_similarities.extend(nr.similarity for _, nr in result.negative_results)
    
    # Calculate average similarities
    avg_positive_similarity = (
        sum(all_positive_similarities) / len(all_positive_similarities)
        if all_positive_similarities else 0.0
    )
    avg_negative_similarity = (
        sum(all_negative_similarities) / len(all_negative_similarities)
        if all_negative_similarities else 0.0
    )
    
    # Calculate results by source type
    results_by_source = defaultdict(lambda: {
        'total': 0,
        'successful': 0,
        'too_short': 0,
        'other_failed': 0,
        'positive_tests_total': 0,
        'positive_tests_passed': 0,
        'negative_tests_total': 0,
        'negative_tests_passed': 0
    })
    
    for result in results.values():
        stats = results_by_source[result.source_type]
        stats['total'] += 1
        
        if result.error is None:
            stats['successful'] += 1
            stats['positive_tests_total'] += len(result.positive_results)
            stats['positive_tests_passed'] += sum(1 for pr in result.positive_results if pr.verified)
            stats['negative_tests_total'] += len(result.negative_results)
            stats['negative_tests_passed'] += sum(1 for _, nr in result.negative_results if not nr.verified)
        elif result.is_too_short:
            stats['too_short'] += 1
        else:
            stats['other_failed'] += 1
    
    return TestSummary(
        total_files=len(results),
        successful_files=successful_files,
        failed_files=other_failed_files + too_short_files,
        total_positive_tests=total_positive_tests,
        successful_positive_tests=successful_positive_tests,
        total_negative_tests=total_negative_tests,
        successful_negative_tests=successful_negative_tests,
        avg_positive_similarity=avg_positive_similarity,
        avg_negative_similarity=avg_negative_similarity,
        results_by_source=dict(results_by_source)
    )

def print_results(results: Dict[str, FileTestResults], summary: TestSummary):
    """Print detailed results and summary."""
    # Print too-short files first
    too_short_files = [(name, result) for name, result in results.items() if result.is_too_short]
    if too_short_files:
        print("\nToo Short Files:")
        print("===============")
        for file_name, result in too_short_files:
            print(f"\nFile: {file_name}")
            print(f"Source Type: {result.source_type}")
            print(f"Duration: {result.file_duration_sec:.1f}s")
            if result.num_segments > 0:
                print(f"Segments possible: {result.num_segments}")
            print(f"Minimum required: {SEGMENT_LENGTH_SEC*2:.1f}s")
            print("-" * 50)
    
    # Print other failed cases
    other_failed = [(name, result) for name, result in results.items() 
                   if result.error is not None and not result.is_too_short]
    if other_failed:
        print("\nOther Failed Cases:")
        print("==================")
        for file_name, result in other_failed:
            print(f"\nFile: {file_name}")
            print(f"Source Type: {result.source_type}")
            print(f"Duration: {result.file_duration_sec:.1f}s")
            print(f"Error: {result.error}")
            print("-" * 50)
    
    print("\nDetailed Results:")
    print("================")
    
    # Group results by source type (only successful files)
    by_source = defaultdict(list)
    for result in results.values():
        if result.error is None:  # Only include successful cases
            by_source[result.source_type].append(result)
    
    # Print results for each source type
    for source_type, source_results in by_source.items():
        print(f"\n{source_type} Files:")
        print("-" * (len(source_type) + 7))
        
        for result in source_results:
            print(f"\n{result.file_name}:")
            print(f"  Registration: Using segment {result.registration_segment_index + 1} of {result.num_verification_segments + 1} total segments")
            
            # Positive results
            pos_success = sum(1 for pr in result.positive_results if pr.verified)
            print(f"  Positive tests: {pos_success}/{len(result.positive_results)} passed")
            for pr in result.positive_results:
                print(f"    Segment {pr.segment_index + 1}: "
                      f"{'✓' if pr.verified else '✗'} "
                      f"(similarity: {pr.similarity:.3f})"
                      f"(segment index: {pr.segment_index})"
                      f"(source segment index: {pr.source_segment_index})")
            
            # Negative results
            neg_success = sum(1 for _, nr in result.negative_results if not nr.verified)
            print(f"  Negative tests: {neg_success}/{len(result.negative_results)} passed")
            for source_file, nr in result.negative_results:
                print(f"    vs {source_file}: "
                      f"{'✗' if not nr.verified else '✓'} "
                      f"(similarity: {nr.similarity:.3f})"
                      f"(segment index: {nr.segment_index})"
                      f"(source segment index: {nr.source_segment_index})")
    
    print("\nSummary:")
    print("========")
    print(f"Total files processed: {summary.total_files}")
    print(f"  - Successful: {summary.successful_files}")
    print(f"  - Too short: {len(too_short_files)}")
    print(f"  - Other failures: {len(other_failed)}")
    
    # File duration analysis
    print("\nFile Duration Analysis:")
    all_durations = [(r.file_name, r.file_duration_sec) for r in results.values()]
    if all_durations:
        avg_duration = sum(d for _, d in all_durations) / len(all_durations)
        min_duration = min((d for _, d in all_durations), default=0)
        max_duration = max((d for _, d in all_durations), default=0)
        print(f"  - Average duration: {avg_duration:.1f}s")
        print(f"  - Minimum duration: {min_duration:.1f}s")
        print(f"  - Maximum duration: {max_duration:.1f}s")
    
    # Short files analysis
    min_required = SEGMENT_LENGTH_SEC * 2
    print(f"\nFiles shorter than required ({min_required:.1f}s):")
    short_files = [(name, duration) for name, duration in all_durations if duration < min_required]
    if short_files:
        for name, duration in short_files:
            print(f"  - {name}: {duration:.1f}s")
    else:
        print("  None")
    
    print("\nTest Results (only from successful files):")
    if summary.total_positive_tests > 0:
        pos_success_rate = (summary.successful_positive_tests/summary.total_positive_tests)*100
        print(f"\nPositive Tests (segments should match):")
        print(f"  - Total tests: {summary.total_positive_tests}")
        print(f"  - Successful matches: {summary.successful_positive_tests}")
        print(f"  - Success rate: {pos_success_rate:.1f}%")
        print(f"  - Average similarity: {summary.avg_positive_similarity:.3f}")
    
    if summary.total_negative_tests > 0:
        neg_success_rate = (summary.successful_negative_tests/summary.total_negative_tests)*100
        print(f"\nNegative Tests (segments should NOT match):")
        print(f"  - Total tests: {summary.total_negative_tests}")
        print(f"  - Successful non-matches: {summary.successful_negative_tests}")
        print(f"  - Success rate: {neg_success_rate:.1f}%")
        print(f"  - Average similarity: {summary.avg_negative_similarity:.3f}")
    
    print("\nResults by source type:")
    for source_type, stats in summary.results_by_source.items():
        print(f"\n{source_type}:")
        print(f"  - Total files: {stats['total']}")
        print(f"  - Successful files: {stats['successful']}")
        print(f"  - Too short files: {stats['too_short']}")
        print(f"  - Other failures: {stats['other_failed']}")
        
        if stats['positive_tests_total'] > 0:
            pos_rate = (stats['positive_tests_passed'] / stats['positive_tests_total']) * 100
            print(f"  - Positive test success rate: {pos_rate:.1f}%")
        
        if stats['negative_tests_total'] > 0:
            neg_rate = (stats['negative_tests_passed'] / stats['negative_tests_total']) * 100
            print(f"  - Negative test success rate: {neg_rate:.1f}%")

# Example of using the system
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Speaker verification test with configurable parameters')
    parser.add_argument('--segment-length', type=float, default=30.0,
                      help='Length of each segment in seconds (default: 30.0)')
    parser.add_argument('--negative-samples', type=int, default=5,
                      help='Number of negative samples to test (default: 5)')
    parser.add_argument('--customer-dir', type=str, default="speaker_segments/customer_segments",
                      help='Directory containing customer audio segments')
    parser.add_argument('--operator-dir', type=str, default="speaker_segments/operator_segments",
                      help='Directory containing operator audio segments')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Similarity threshold for verification (default: 0.7)')
    
    args = parser.parse_args()
    
    # Configuration parameters
    SEGMENT_LENGTH_SEC = args.segment_length
    NUM_NEGATIVE_SAMPLES = args.negative_samples
    
    print(f"Using segment length: {SEGMENT_LENGTH_SEC} seconds")
    print(f"Using negative samples: {NUM_NEGATIVE_SAMPLES}")
    
    # Initialize model
    print("Loading model...")
    model = load_model()

    # Initialize a dictionary to store user embeddings
    embeddings_db = {}

    # Get files from both directories
    customer_dir = args.customer_dir
    operator_dir = args.operator_dir
    
    # Collect all WAV files from both directories with their full paths
    wav_files = []
    for directory in [customer_dir, operator_dir]:
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith('.WAV') or f.endswith('.wav')]
            wav_files.extend([(f, os.path.join(directory, f)) for f in files])
    
    print(f"\nFound {len(wav_files)} WAV files to process.")
    
    results = {}
    # Pass the list of all filenames for random selection
    all_filenames = [f[0] for f in wav_files]
    
    # Process all files
    for filename, filepath in wav_files:
        results[filename] = process_audio_file(
            filepath, model, embeddings_db, all_filenames,
            segment_length_sec=SEGMENT_LENGTH_SEC,
            num_negative_samples=NUM_NEGATIVE_SAMPLES
        )
    
    # Calculate summary statistics
    summary = calculate_summary(results)
    
    # Print all results
    print_results(results, summary)
