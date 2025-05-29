import json

# Load source notebook
with open('source_notebook.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract all cells
cells = data['cells']

# Optionally, write to a new notebook
new_notebook = {
    "cells": cells,
    "metadata": data["metadata"],
    "nbformat": data["nbformat"],
    "nbformat_minor": data["nbformat_minor"]
}

with open('./fraud_detect_model2.5.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_notebook, f, indent=2)
