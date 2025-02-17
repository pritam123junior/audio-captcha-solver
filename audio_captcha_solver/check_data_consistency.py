# check_data_consistency.py
import os

audio_dir = "data/audio"  # Or your actual audio directory
image_dir = "data/images"  # Or your actual image directory

print(f"Audio directory: {audio_dir}")
print(f"Image directory: {image_dir}")

if not os.path.exists(audio_dir):
    print(f"ERROR: Audio directory does not exist: {audio_dir}")
    exit()

if not os.path.exists(image_dir):
    print(f"ERROR: Image directory does not exist: {image_dir}")
    exit()

audio_files = sorted([f[:-4] for f in os.listdir(audio_dir) if f.endswith(".wav") and not f.startswith(".")])
image_files = sorted([f[:-4] for f in os.listdir(image_dir) if f.endswith(".png") and not f.startswith(".")])

print(f"Audio files: {audio_files}")
print(f"Image files: {image_files}")


if audio_files == image_files:
    print("Filenames match!")
else:
    print("Filenames do NOT match!")

    # Find files present only in audio
    audio_only = set(audio_files) - set(image_files)
    if audio_only:
        print("Files only in audio:")
        for file in sorted(audio_only):
            print(f"  {file}.wav")
            file_path = os.path.join(audio_dir, f"{file}.wav")
            # os.remove(file_path)  # Uncomment to DELETE audio files AFTER careful testing
            print(f"  (Would be deleted): {file_path}")  # Safety measure

    # Find files present only in images
    image_only = set(image_files) - set(audio_files)
    if image_only:
        print("Files only in images:")
        for file in sorted(image_only):
            print(f"  {file}.png")
            file_path = os.path.join(image_dir, f"{file}.png")
            # os.remove(file_path)  # Uncomment to DELETE image files AFTER careful testing
            print(f"  (Would be deleted): {file_path}")  # Safety measure

    # Find common files but with different names (e.g., case differences)
    common_files = set(audio_files) & set(image_files)
    different_names = []
    for file in common_files:
        if file + ".wav" not in os.listdir(audio_dir) or file + ".png" not in os.listdir(image_dir):
            different_names.append(file)
    if different_names:
        print("Common files with different names:")
        for file in different_names:
            print(f"  {file}")
            audio_path = os.path.join(audio_dir, f"{file}.wav")
            image_path = os.path.join(image_dir, f"{file}.png")

            # Check if only the case is different
            if os.path.exists(audio_path.lower()) and os.path.exists(image_path.lower()):
                print(f"  Rename {audio_path.lower()} to {audio_path}")
                os.rename(audio_path.lower(), audio_path)
                print(f"  Rename {image_path.lower()} to {image_path}")
                os.rename(image_path.lower(), image_path)
            else: # If not just case difference, then delete
                 os.remove(audio_path) # Uncomment to DELETE (If not just case difference)
                 os.remove(image_path) # Uncomment to DELETE (If not just case difference)
              