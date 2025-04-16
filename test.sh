#!/bin/bash
# Automatic testing of the auto-grader script for all of the course
# Author : Philippe Esling

# Check if one argument is passed
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <index>"
  exit 1
fi

# Input index
index="$1"

# Define an array of file paths (edit to your real filenames)
files=(
    "00_setup.ipynb"
    "01a_machine_learning_autograde.ipynb"
    "02_neural_networks.ipynb"
    "03_advanced_networks.ipynb"
    "04_deep_learning.ipynb"
    "05_probabilities_bayesian.ipynb"
    "06_latent_models.ipynb"
    "07_approximate_inference.ipynb"
    "08a_variational_auto_encoders.ipynb"
    "08b_normalizing_flows.ipynb"
    "09_adversarial_learning.ipynb"
    "10_diffusion_models.ipynb"
)
tests=(
    "test_cml_00"
    "test_cml_01"
)

# Check if the index is valid
if [ "$index" -lt 0 ] || [ "$index" -ge "${#files[@]}" ]; then
  echo "Index out of range. Valid range: 0 to $((${#files[@]} - 1))"
  exit 1
fi

# Selected file
selected_file="${files[$index]}"
basename="${tests[$index]}"
# Remove the Python cache
rm -rf tests/__pycache__ 
echo "Selected file: $selected_file"

# Step 1: Run a command on the selected file (edit this command)
echo " - Running jupyter nbconvert command."
jupyter nbconvert --to script "$selected_file" --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="noexport" --output tests/assignment

# Step 2: Look for files with same basename in subfolders
echo " - Looking for matching tests with $basename"
matches=$(find ./tests -type f -name "*$basename*")
echo $matches
for match in $matches; do
  echo "    . Found: $match"
  echo "    . Running pytest"
  pytest "$match" 
done
