import os
import zipfile

def package_repository():
    project_dir = os.path.dirname(__file__)
    zip_filename = os.path.join(project_dir, "workflow_centric_medical_ai.zip")

    # Files & directories to include
    include_files = [
        "README.md",
        "DATASET_SETUP.md",
        "requirements.txt",
        "prepare_dataset.py",
        "evaluate_pipeline.py",
        "main.py",
        "test_step1.py",
        "test_step2.py",
        "test_step3.py",
        "test_step4.py"
    ]

    include_dirs = ["src", "models"]

    print("--------------------------------------------------")
    print("[ZIP] Packaging GitHub-Ready Repository into ZIP")
    print("--------------------------------------------------")

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add root files
        for filename in include_files:
            file_path = os.path.join(project_dir, filename)
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=os.path.join("workflow_centric_medical_ai", filename))
                print(f"  + Added file: {filename}")

        # Add directories recursively
        for dir_name in include_dirs:
            dir_path = os.path.join(project_dir, dir_name)
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        if file.endswith((".pyc", ".pyo")):
                            continue
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, project_dir)
                        arcname = os.path.join("workflow_centric_medical_ai", rel_path)
                        zipf.write(full_path, arcname=arcname)
                        print(f"  + Added dir file: {rel_path}")

    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print("--------------------------------------------------")
    print(f"[OK] SUCCESS! Created ZIP Archive at:")
    print(f"   {zip_filename} ({zip_size_mb:.2f} MB)")
    print("--------------------------------------------------")
    return zip_filename

if __name__ == "__main__":
    package_repository()
