import os
import glob
import zipfile
import tarfile

def main():
    base_dir = "./eo4wildfires_local"
    extract_dir = "./eo4wildfires_local/extracted"

    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist. Run the download script first.")
        return

    # 1. Scan downloaded files
    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_dir) for f in filenames]
    print(f"Found {len(all_files)} total files in '{base_dir}':")
    for f in all_files[:10]:
        print(f"  - {f}")

    # 2. Extract .tar.gz or .zip archives if present
    os.makedirs(extract_dir, exist_ok=True)
    for file_path in all_files:
        if file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
            print(f"\nExtracting TAR archive: {file_path}...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            print("TAR extraction complete.")

        elif file_path.endswith(".zip"):
            print(f"\nExtracting ZIP archive: {file_path}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("ZIP extraction complete.")

    # 3. Check for ready-to-use array files
    data_files = glob.glob(os.path.join(base_dir, "**/*.npz"), recursive=True) + \
                 glob.glob(os.path.join(base_dir, "**/*.npy"), recursive=True)
    print(f"\nTotal .npz / .npy data files found: {len(data_files)}")

if __name__ == "__main__":
    main()