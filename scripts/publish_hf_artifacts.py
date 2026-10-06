"""Publish safe public experiment artifacts to Hugging Face Hub (MIHMahmudEli/david-net-av-v2)."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi
from src.scheduler.config import hf_token

def main():
    token = hf_token()
    if not token:
        print("ERROR: Hugging Face token not found in .env")
        sys.exit(1)
        
    api = HfApi(token=token)
    repo_id = "MIHMahmudEli/david-net-av-v2"
    repo_type = "model"
    
    print(f"Creating/verifying repo: {repo_id} (type={repo_type}, public)...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False, exist_ok=True)
    
    # Check if HF generated a default README.md on repo creation
    remote_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    print(f"Initial remote files in {repo_id}: {remote_files}")
    for rf in remote_files:
        if rf.lower().endswith(".md") or rf.lower() == "readme.md":
            print(f"Deleting default remote file: {rf}")
            api.delete_file(path_in_repo=rf, repo_id=repo_id, repo_type=repo_type, commit_message=f"Remove default {rf}")
            
    # Staging directory verification
    staging_dir = Path("hf_artifacts_staging")
    if not staging_dir.exists():
        print(f"ERROR: {staging_dir} does not exist")
        sys.exit(1)
        
    forbidden_exts = (
        '.md', '.markdown', '.mdown', '.mkd', '.txt', '.text',
        '.pdf', '.tex', '.latex', '.doc', '.docx', '.odt', '.rtf',
        '.rst', '.html', '.htm', '.epub', '.bib', '.bibtex',
        '.ppt', '.pptx', '.odp', '.pages', '.rtfd'
    )
    
    staged_files = [p for p in staging_dir.rglob("*") if p.is_file()]
    for p in staged_files:
        if any(p.name.lower().endswith(ext) for ext in forbidden_exts):
            print(f"BLOCKLIST VIOLATION: {p}")
            sys.exit(1)
            
    print(f"Uploading {len(staged_files)} safe artifacts from {staging_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Release public experiment artifacts, metrics, and models",
    )
    
    # Final remote verification
    final_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    print(f"Total remote files after upload: {len(final_files)}")
    
    violations = [f for f in final_files if any(f.lower().endswith(ext) for ext in forbidden_exts)]
    if violations:
        print(f"ERROR: Remote violations found: {violations}")
        for v in violations:
            api.delete_file(path_in_repo=v, repo_id=repo_id, repo_type=repo_type, commit_message=f"Purge forbidden file {v}")
        final_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        violations = [f for f in final_files if any(f.lower().endswith(ext) for ext in forbidden_exts)]
        if violations:
            print("FATAL: Still has violations after purge!")
            sys.exit(1)
            
    print("VERIFICATION SUCCESSFUL: 0 forbidden document extensions exist remotely!")
    print(f"Remote file tree ({len(final_files)} items):")
    for f in sorted(final_files):
        print(f"  {f}")

if __name__ == "__main__":
    main()
