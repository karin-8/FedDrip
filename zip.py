import os
import zipfile
from tqdm import tqdm

def zip_files_and_folders(source_dir, output_zip):
    """
    Zips files and folders in the source directory, ensuring no single file exceeds 1MB.

    :param source_dir: Path to the source directory
    :param output_zip: Path to the output ZIP file
    """
    max_size = 1 * 1024 * 1024  # 1MB in bytes

    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in tqdm(files):
                if file.endswith('.py') or file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    if os.path.getsize(file_path) <= max_size:
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname)
            # for dir_name in tqdm(dirs):
            #     dir_path = os.path.join(root, dir_name)
            #     dir_size = get_folder_size(dir_path)
            #     if dir_size <= max_size:
            #         add_folder_to_zip(zipf, dir_path, source_dir)


def get_folder_size(folder):
    """Calculate the total size of a folder and its contents."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def add_folder_to_zip(zipf, folder, source_dir):
    """Add a folder and its contents to the zip file."""
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.py') or file.endswith('.json'):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    source_directory = "./"  # Replace with your source directory
    output_zip_file = "output.zip"  # Replace with your desired output ZIP file name

    zip_files_and_folders(source_directory, output_zip_file)
    print(f"Files and folders zipped successfully into {output_zip_file}!")
