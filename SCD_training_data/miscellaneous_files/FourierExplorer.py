import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, Button

# Function to calculate and display the Fourier transform
def calculate_fourier_transform(image_path, ax):
    # Load the image
    image = plt.imread(image_path)
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        image = np.mean(image, axis=-1)
    
    # Compute the 2D Fourier transform of the image
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.abs(fft_shifted)

    # Display the magnitude spectrum
    ax.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    ax.set_title(f'Fourier Transform: {image_path}')

# Function to handle file selection
def select_files():
    file_paths = filedialog.askopenfilenames(title='Select up to five image files', filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.gif;*.bmp')])
    if file_paths:
        display_fourier_transforms(file_paths)

# Function to display Fourier transforms in the same window
def display_fourier_transforms(file_paths):
    # Create the main tkinter window
    root = tk.Tk()
    root.title('Fourier Transform Viewer')

    # Create a matplotlib figure and subplots within the tkinter window
    num_files = min(len(file_paths), 5)
    fig, axes = plt.subplots(1, num_files, figsize=(4 * num_files, 4))
    canvas_widget = FigureCanvasTkAgg(fig, master=root)
    canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Create buttons to trigger the Fourier transform calculation for each file
    for i in range(num_files):
        ax = axes[i] if num_files > 1 else axes
        ax.axis('off')  # Turn off axis labels and ticks
        calculate_fourier_transform(file_paths[i], ax)

    # Set the aspect ratio of the plots to be equal
    for ax in axes:
        ax.set_aspect('equal', 'box')

    # Create a button to trigger the file selection
    select_button = Button(root, text='Select Files', command=select_files)
    select_button.pack()

    # Start the tkinter main loop
    root.mainloop()

# Initial file selection
select_files()


