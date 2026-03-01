#!/usr/bin/env python3
"""Minimal checksum verifier."""
import json
import hashlib
import sys

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify():
    try:
        with open('checksums.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: checksums.json not found")
        sys.exit(1)
    
    print(f"Verifying checksums from {data['timestamp']}")
    print("=" * 60)
    
    all_good = True
    
    # Check repo files
    for file, expected in data['repository_files'].items():
        try:
            actual = sha256_file(file)
            if actual == expected:
                print(f"✓ {file}")
            else:
                print(f"✗ {file} (MISMATCH)")
                all_good = False
        except FileNotFoundError:
            print(f"✗ {file} (MISSING)")
            all_good = False

if __name__ == '__main__':
    sys.exit(verify())
