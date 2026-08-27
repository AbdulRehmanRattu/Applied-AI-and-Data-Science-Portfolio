import numpy as np

def read_labels(label_file):
    """
    Reads and processes label data from a file.
    
    Args:
    label_file (str): The path to the file containing labels.
    
    Returns:
    tuple: A tuple containing a list of labels and the number of labels.
    """
    # Open the file and read lines
    f = open(label_file)
    lines = f.readlines()
    
    # Strip whitespace and convert each line to an integer
    labels = [int(item.strip()) for item in lines]
    
    # Calculate the number of labels
    sample_num = len(labels)
    
    return labels, sample_num

def load_sample(sample_file, sample_num):
    """
    Loads and processes sample data from a file, converting text representations of images
    into NumPy arrays.
    
    Args:
    sample_file (str): The path to the file containing image data.
    sample_num (int): The number of samples to process.
    
    Returns:
    list: A list of 2D NumPy arrays representing images.
    """
    # Open the file and read lines
    f = open(sample_file)
    lines = f.readlines()
    
    # Calculate file dimensions
    file_length = int(len(lines))  # Total number of lines in the file
    width = int(len(lines[0]))     # Width of the image (characters per line)
    length = int(file_length / sample_num)  # Height of each image
    
    # Debug print statements to check image dimensions
    print(len(lines[0]), file_length / sample_num)
    print(width, length)
    
    all_image = []  # List to hold all image data
    
    # Process each image
    for i in range(sample_num):
        # Create an array of zeros for each image based on calculated dimensions
        single_image = np.zeros((length, width))
        count = 0
        
        # Iterate through each line that makes up the current image
        for j in range(length * i, length * (i + 1)):
            single_line = lines[j]
            
            # Check each character in the line and set pixel values
            for k in range(len(single_line)):
                if single_line[k] in ["+", "#"]:
                    single_image[count, k] = 1
            count += 1
        
        # Add the processed image to the list
        all_image.append(single_image)
    
    return all_image
