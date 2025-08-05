import joblib
import torch
import numpy as np
import os
import pickle

# Create output directory if it doesn't exist
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

# Define base paths
BASE_PATH = 'MFMC-Demo'
DATA_DIR = f'{BASE_PATH}/Data_processed'

create_folder_if_not_exists(DATA_DIR)

# Define channels for each modality
EEG_CHANNELS = list(range(0, 32))  # 32 EEG channels
EOG_CHANNELS = [32, 33]  # 2 EOG channels
TEMP_CHANNELS = [39]  # Temperature channel

# Define windowing parameters
SAMPLE_RATE = 128  # Hz
WINDOW_SIZE = 10 * SAMPLE_RATE  # 10 seconds
WINDOW_OFFSET = 3 * SAMPLE_RATE  # Skip first 3 seconds
WINDOW_STRIDE = int(0.4 * SAMPLE_RATE)  # 0.4 second stride

# Change this to True to enable outlier filtering
FILTER_OUTLIERS = True

def load_deap_data():
    """Load the DEAP dataset files."""
    print("Loading DEAP dataset...")
    
    all_eeg_data = []
    all_eog_data = []
    all_temp_data = []
    all_valence = []
    all_arousal = []
    all_subject_ids = []
    
    # Load data for each subject (1-32)
    for subject_id in range(1, 33):
        # Adjust filename based on DEAP dataset structure
        filename = f'{BASE_PATH}/DEAP/s{subject_id:02d}.dat'
        
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                
            # Extract data
            # DEAP structure: data['data'] has shape (40 trials, 40 channels, 8064 samples)
            # Each trial is 63 seconds (8064 samples at 128Hz)
            # Labels are in data['labels'] as (40 trials, 4) - valence, arousal, dominance, liking
            
            for trial in range(len(data['data'])):
                trial_data = data['data'][trial]
                
                # Extract EEG channels
                eeg_trial = trial_data[EEG_CHANNELS, :]
                
                # Extract EOG channels
                eog_trial = trial_data[EOG_CHANNELS, :]
                
                # Extract TEMP channels
                temp_trial = trial_data[TEMP_CHANNELS, :]
                
                # Process with windowing
                for start_idx in range(WINDOW_OFFSET, trial_data.shape[1] - WINDOW_SIZE + 1, WINDOW_STRIDE):
                    end_idx = start_idx + WINDOW_SIZE
                    
                    eeg_window = eeg_trial[:, start_idx:end_idx]
                    eog_window = eog_trial[:, start_idx:end_idx]
                    temp_window = temp_trial[:, start_idx:end_idx]
                    
                    all_eeg_data.append(eeg_window)
                    all_eog_data.append(eog_window)
                    all_temp_data.append(temp_window)
                    
                    # Get emotion labels (valence and arousal)
                    valence = data['labels'][trial, 0]  # Valence
                    arousal = data['labels'][trial, 1]  # Arousal
                    
                    # Convert to binary categories (high/low)
                    valence_label = 1 if valence > 5 else 0
                    arousal_label = 1 if arousal > 5 else 0
                    
                    all_valence.append(valence_label)
                    all_arousal.append(arousal_label)
                    all_subject_ids.append(subject_id - 1)  # Zero-indexed
        
        except Exception as e:
            print(f"Error loading subject {subject_id}: {e}")
    
    # Convert lists to numpy arrays
    eeg_data = np.stack(all_eeg_data)
    eog_data = np.stack(all_eog_data)
    temp_data = np.stack(all_temp_data)
    valence = np.array(all_valence)
    arousal = np.array(all_arousal)
    subject_ids = np.array(all_subject_ids)
    
    # Combine valence and arousal to create 4 emotion quadrants
    # (0,0): low valence, low arousal - sad
    # (0,1): low valence, high arousal - fear/angry
    # (1,0): high valence, low arousal - calm/relaxed
    # (1,1): high valence, high arousal - happy/excited
    emotion_labels = valence * 2 + arousal
    
    return eeg_data, eog_data, temp_data, emotion_labels, subject_ids, valence, arousal

def preprocess_data(eeg_data, eog_data, temp_data, emotion_labels, subject_ids, valence, arousal, filter_outliers=True):
    """Apply normalization and optionally remove outliers."""
    print("Preprocessing data...")
    
    # Convert to torch tensors
    eeg_data = torch.from_numpy(eeg_data).float()
    eog_data = torch.from_numpy(eog_data).float()
    temp_data = torch.from_numpy(temp_data).float()
    emotion_labels = torch.from_numpy(emotion_labels)
    subject_ids = torch.from_numpy(subject_ids)
    valence = torch.from_numpy(valence)
    arousal = torch.from_numpy(arousal)
    
    # Normalization: Normalize signals by the magnitude of the first sample
    eeg_max = eeg_data[0].abs().max()
    eog_max = eog_data[0].abs().max()
    temp_max = temp_data[0].abs().max()
    
    eeg_data = eeg_data / eeg_max
    eog_data = eog_data / eog_max
    temp_data = temp_data / temp_max
    
    if filter_outliers:
        print("Applying outlier filtering...")
        # Remove outlier samples that have magnitude larger than 5 or smaller than -5
        
        # Check EEG data for outliers > 5
        if_keep = (eeg_data > 5).sum(1).sum(1)
        valid_samples = torch.where(if_keep < 1)[0]
        eeg_data = eeg_data[valid_samples]
        eog_data = eog_data[valid_samples]
        temp_data = temp_data[valid_samples]
        emotion_labels = emotion_labels[valid_samples]
        subject_ids = subject_ids[valid_samples]
        valence = valence[valid_samples]
        arousal = arousal[valid_samples]
        
        # Check EEG data for outliers < -5
        if_keep = (eeg_data < -5).sum(1).sum(1)
        valid_samples = torch.where(if_keep < 1)[0]
        eeg_data = eeg_data[valid_samples]
        eog_data = eog_data[valid_samples]
        temp_data = temp_data[valid_samples]
        emotion_labels = emotion_labels[valid_samples]
        subject_ids = subject_ids[valid_samples]
        valence = valence[valid_samples]
        arousal = arousal[valid_samples]
        
        # Check EOG data for outliers > 5
        if_keep = (eog_data > 5).sum(1).sum(1)
        valid_samples = torch.where(if_keep < 1)[0]
        eeg_data = eeg_data[valid_samples]
        eog_data = eog_data[valid_samples]
        temp_data = temp_data[valid_samples]
        emotion_labels = emotion_labels[valid_samples]
        subject_ids = subject_ids[valid_samples]
        valence = valence[valid_samples]
        arousal = arousal[valid_samples]
        
        # Check EOG data for outliers < -5
        if_keep = (eog_data < -5).sum(1).sum(1)
        valid_samples = torch.where(if_keep < 1)[0]
        eeg_data = eeg_data[valid_samples]
        eog_data = eog_data[valid_samples]
        temp_data = temp_data[valid_samples]
        emotion_labels = emotion_labels[valid_samples]
        subject_ids = subject_ids[valid_samples]
        valence = valence[valid_samples]
        arousal = arousal[valid_samples]
        
        # Check TEMP data for outliers > 5
        if_keep = (temp_data > 5).sum(1).sum(1)
        valid_samples = torch.where(if_keep < 1)[0]
        eeg_data = eeg_data[valid_samples]
        eog_data = eog_data[valid_samples]
        temp_data = temp_data[valid_samples]
        emotion_labels = emotion_labels[valid_samples]
        subject_ids = subject_ids[valid_samples]
        valence = valence[valid_samples]
        arousal = arousal[valid_samples]
        
        # Check TEMP data for outliers < -5
        if_keep = (temp_data < -5).sum(1).sum(1)
        valid_samples = torch.where(if_keep < 1)[0]
        eeg_data_filtered = eeg_data[valid_samples]
        eog_data_filtered = eog_data[valid_samples]
        temp_data_filtered = temp_data[valid_samples]
        emotion_labels_filtered = emotion_labels[valid_samples]
        subject_ids_filtered = subject_ids[valid_samples]
        valence_filtered = valence[valid_samples]
        arousal_filtered = arousal[valid_samples]
    else:
        print("Skipping outlier filtering...")
        # Keep all samples
        eeg_data_filtered = eeg_data
        eog_data_filtered = eog_data
        temp_data_filtered = temp_data
        emotion_labels_filtered = emotion_labels
        subject_ids_filtered = subject_ids
        valence_filtered = valence
        arousal_filtered = arousal
    
    return eeg_data_filtered, eog_data_filtered, temp_data_filtered, emotion_labels_filtered, subject_ids_filtered, valence_filtered, arousal_filtered

# Main execution
if __name__ == "__main__":
    print("Starting preprocessing for DEAP dataset...")
    
    print(f"Outlier filtering: {'ENABLED' if FILTER_OUTLIERS else 'DISABLED'}")
    
    # Load and window the data
    eeg_data, eog_data, temp_data, emotion_labels, subject_ids, valence, arousal = load_deap_data()
    
    # Preprocess the windowed data
    eeg_data_filtered, eog_data_filtered, temp_data_filtered, emotion_labels_filtered, subject_ids_filtered, valence_filtered, arousal_filtered = preprocess_data(
        eeg_data, eog_data, temp_data, emotion_labels, subject_ids, valence, arousal, filter_outliers=FILTER_OUTLIERS
    )
    
    # Save processed data
    print(f"Saving {len(eeg_data_filtered)} valid samples...")
    np.save(f'{DATA_DIR}/subject.npy', subject_ids_filtered)
    np.save(f'{DATA_DIR}/emotion_labels.npy', emotion_labels_filtered)
    np.save(f'{DATA_DIR}/valence.npy', valence_filtered)
    np.save(f'{DATA_DIR}/arousal.npy', arousal_filtered)
    np.save(f'{DATA_DIR}/eeg_data.npy', eeg_data_filtered.numpy())
    np.save(f'{DATA_DIR}/eog_data.npy', eog_data_filtered.numpy())
    np.save(f'{DATA_DIR}/temp_data.npy', temp_data_filtered.numpy())
    
    print("Preprocessing complete!")
    print(f"EEG data shape: {eeg_data_filtered.shape}")
    print(f"EOG data shape: {eog_data_filtered.shape}")
    print(f"TEMP data shape: {temp_data_filtered.shape}")
    print(f"Emotion labels shape: {emotion_labels_filtered.shape}")
    
    # Summary of subjects in processed dataset
    unique_subjects = np.unique(subject_ids_filtered)
    total_subjects_kept = len(unique_subjects)
    original_subjects = 32  # DEAP has 32 subjects
    
    # Convert zero-indexed back to one-indexed for display
    subject_ids_display = sorted(unique_subjects + 1)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Original subjects in DEAP dataset: {original_subjects}")
    print(f"Subjects kept in processed dataset: {total_subjects_kept}")
    print(f"Subjects removed: {original_subjects - total_subjects_kept}")
    print(f"Subject retention rate: {(total_subjects_kept/original_subjects)*100:.1f}%")
    print(f"Total samples after preprocessing: {len(eeg_data_filtered)}")
    print(f"Average samples per subject: {len(eeg_data_filtered)/total_subjects_kept:.1f}")
    print("\nSubject IDs in processed dataset:")
    print(f"  {subject_ids_display}")
    
    # Show which subjects were removed (if any)
    if total_subjects_kept < original_subjects:
        all_original_subjects = list(range(1, original_subjects + 1))
        removed_subjects = [s for s in all_original_subjects if s not in subject_ids_display]
        print(f"\nRemoved subject IDs:")
        print(f"  {removed_subjects}")
    print("="*50)
