#!/usr/bin/env python3
"""Minimal checksum generator"""
import hashlib
import json
from pathlib import Path
from datetime import datetime

def sha256_file(filepath):
    """Get SHA256 of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def generate():
    """Generate checksums for all files."""
    checksums = {}
    
    # Skip these
    skip = {'.git', '__pycache__', '.DS_Store', 'Thumbs.db', '*.pyc'}
    
    for path in Path('.').rglob('*'):
        if path.is_file():
            # Skip ignored paths
            if any(pattern in str(path) for pattern in skip):
                continue
            
            # Skip checksum files themselves
            if path.name in ['checksums.json', 'CHECKSUMS.md']:
                continue
            
            checksums[str(path)] = sha256_file(path)
    
    # Save JSON
    with open('checksums.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'repository_files': checksums
        }, f, indent=2)
    
    # Save human-readable
    with open('CHECKSUMS.md', 'w') as f:
        f.write(f"# Checksums - Generated {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## Zenodo Models\n```\n")
        f.write("```\n\n## Repository Files\n```\n")
        for file, h in sorted(checksums.items()):
            f.write(f"{h}  {file}\n")
        f.write("```\n")
    
    print(f"Generated checksums for {len(checksums)} files")

if __name__ == '__main__':
    generate()
