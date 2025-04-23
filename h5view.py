import h5py

# Open the .h5 file
file_path = r"/home/adimundada/CheXzero/train_data/cxr.h5"  # Replace with the path to your .h5 file
with h5py.File(file_path, 'r') as file:
    # Print the keys of the file (top-level groups or datasets)
    print("Keys in the HDF5 file:", list(file.keys()))
    print(file['cxr'])

import matplotlib.pyplot as plt
with h5py.File(file_path, 'r') as file:
    # Access the dataset
    dataset = file['cxr']  # 'cxr' is the name of the dataset

    # Get the first image (index 0)
    first_image = dataset[1]  # Shape (320, 320)

    # Display the first image using matplotlib
    plt.imshow(first_image, cmap='gray')  # Assuming it's a grayscale image
    plt.title("First Image")
    plt.axis('off')  # Hide the axes for a cleaner view
    plt.show()

