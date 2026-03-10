#!/usr/bin/env python3
"""
Script to remove author information from Python files
"""

import os
import re

# Pattern 1: Copied from RT-DETR with lyuwenyu copyright
pattern1 = r'"""\nCopied from RT-DETR \(https://github\.com/lyuwenyu/RT-DETR\)\nCopyright\(c\) 2023 lyuwenyu\. All Rights Reserved\.\n"""'

# Pattern 2: Copied from RT-DETR with Copyright(c) format
pattern2 = r'"""\nCopied from RT-DETR \(https://github\.com/lyuwenyu/RT-DETR\)\nCopyright\(c\) 2023 lyuwenyu\. All Rights Reserved\.\n"""'

# List of files to process based on grep results
files_to_process = [
    'engine/core/yaml_utils.py',
    'engine/data/__init__.py',
    'engine/data/_misc.py',
    'engine/backbone/__init__.py',
    'engine/misc/__init__.py',
    'engine/misc/visualizer.py',
    'tools/inference/openvino_inf.py',
    'engine/backbone/utils.py',
    'engine/backbone/torchvision_model.py',
    'engine/solver/__init__.py',
    'engine/solver/clas_engine.py',
    'engine/solver/clas_solver.py',
    'engine/optim/__init__.py',
    'engine/optim/warmup.py',
]

def clean_file(filepath):
    """Remove author information from a Python file"""   try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Handle the specific pattern with lyuwenyu
        if 'lyuwenyu' in content and content.startswith('"""'):
            # Find the closing """ after the copyright line
            match = re.search(r'"""\nCopied from RT-DETR.*?Copyright.*?lyuwenyu.*?\n"""', content, re.DOTALL)
            if match:
                # Replace with a generic docstring
                replacement = '"""Core module"""'
                content = content[:match.start()] + replacement + content[match.end():]
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function"""
    processed = 0
    for filepath in files_to_process:
        full_path = os.path.join('.', filepath)
        if os.path.exists(full_path):
            if clean_file(full_path):
                print(f"Cleaned: {filepath}")
                processed += 1
        else:
            print(f"File not found: {filepath}")
    
    print(f"\nTotal files cleaned: {processed}")

if __name__ == '__main__':
    main()
