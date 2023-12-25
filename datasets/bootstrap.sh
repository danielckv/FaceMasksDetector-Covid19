# Create directories for test, train, and eval
mkdir -p test train eval

# Download the dataset
wget -O dataset.zip https://example.com/dataset.zip

# Unzip the dataset
unzip dataset.zip

# Move the images to the respective directories
mv dataset/test/* test/
mv dataset/train/* train/
mv dataset/eval/* eval/

# Clean up
rm -rf dataset.zip dataset
