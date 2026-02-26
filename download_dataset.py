#!/usr/bin/env python3
"""
Download the LUNA16 dataset using kagglehub.
This dataset is used for lung cancer detection research.
"""

import kagglehub

def main():
    print("Downloading LUNA16 dataset...")
    print("Note: This dataset is ~60GB and may take a while to download.")
    
    # Download latest version of the LUNA16 dataset
    path = kagglehub.dataset_download("avc0706/luna16")
    
    print(f"\n✓ Download complete!")
    print(f"Path to dataset files: {path}")
    
    return path

if __name__ == "__main__":
    main()
