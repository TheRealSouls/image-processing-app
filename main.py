import cv2
import helpers
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os

# Load image once at the start
while True:
    image_path = input("Enter path to image file: ").strip()
    if not os.path.isfile(image_path):
        print("File not found. Try again.")
        continue
    try:
        image_array = img.imread(image_path)
        break
    except Exception as e:
        print(f"Invalid image file: {e}")


image_array = img.imread(image_path)
choice = "0"

while choice != "10":
    # Display all of the available options
    print("Welcome to the image processing toolkit!")
    print("Would you like to:")
    print("1. Display the image")
    print("2. Convert to grayscale")
    print("3. Blur the image")
    print("4. Edge detection")
    print("5. Sharpen the image")
    print("6. Fix bad lighting (histogram equalisation)")
    print("7. Save Image")
    print("8. Reset Image")
    print("9. Import Image")
    print("10. Exit ")

    choice = input("Select your option: ")
    
    if choice == "1":
        if image_array.ndim == 3:
            plt.imshow(image_array)
        else:
            plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        plt.show()
    elif choice == "2":
        if image_array.ndim == 3:
            image_array = helpers.apply_grayscale(image_array)
            plt.imshow(image_array, cmap='gray')
            plt.axis('off')
            plt.show()
            print("Image converted to grayscale")
        else:
            print("Image is already grayscale")

    elif choice == "3":
        # Ask user for blur strength from 1 (slight) to 10 (strong)
        while True:
            try:
                strength = int(input("Choose blur strength (1 = slight, 10 = very strong): "))
                if 1 <= strength <= 10:
                    break
                else:
                    print("Please enter a number between 1 and 10.")
            except ValueError:
                print("Please enter a valid integer between 1 and 10.")

        # Map strength to kernel size and sigma
        # Kernel sizes: 3,5,7,...,21 for strengths 1..10
        kernel_size = 3 + 2 * (strength - 1)
        sigma = strength  # higher strength -> larger sigma

        kernel = helpers.gaussian_kernel(size=kernel_size, sigma=sigma)
        image_array = helpers.apply_gaussian_blur(image_array, kernel)

        # Let matplotlib handle dtype/range: float images in [0,1] or [0,255] work directly
        if image_array.ndim == 3:
            plt.imshow(image_array)
        else:
            plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        plt.show()

        print(f"Gaussian blur applied with strength {strength} (kernel size {kernel_size}, sigma {sigma})")

    elif choice == "4":
        # Convert to grayscale if needed
        if image_array.ndim == 3:
            image_array = helpers.apply_grayscale(image_array)
        
        # Convert to uint8 format for cv2.Sobel
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)

        gx = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx**2 + gy**2)

        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        image_array = magnitude.astype(np.uint8)

        plt.imshow(image_array, cmap='gray')
        plt.axis('off')
        plt.show()
        
        print("Edge detection applied")
    elif choice == "5":
        # Normalize image to 0-255 range if needed
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.float64)
        
        # Apply sharpening (preserves colors for color images)
        image_array = helpers.sharpen_image(image_array)
        
        # Display the sharpened image
        if image_array.ndim == 3:
            plt.imshow(image_array / 255.0)  # Normalize for display if needed
        else:
            plt.imshow(image_array, cmap="gray")
        plt.axis("off")
        plt.show()
        
        print("Image sharpening applied")

    elif choice == "6":
        # Fix bad lighting (histogram equalisation)
        image_array = helpers.histogram_equalisation(image_array)
        
        # Display the equalized image
        if image_array.ndim == 3:
            plt.imshow(image_array / 255.0)  # Normalize for display
        else:
            plt.imshow(image_array, cmap="gray")
        plt.axis("off")
        plt.show()
        
        print("Histogram equalisation applied")
    elif choice == "7":
        save_img = helpers.to_uint8(image_array)

        filename = input("Enter filename (to save as): ")

        if filename == "":
            print("File saving cancelled.")
            continue

        if "." not in filename:
            filename += ".png"
        
        save_dir = "saves"
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, filename)

        if save_img.ndim == 2:
            cv2.imwrite(filepath, save_img)
        else:
            cv2.imwrite(filepath, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))

        print(f"Image saved as {filename}.\nPlease check the saves folder for your image.")
    elif choice == "8":
        # Reset to the current base image (which may be an imported one)
        image_array = img.imread(image_path)
        print("Image reset successfully")
    elif choice == "9":
        # Import a new image from disk
        path = input("Enter path to image file: ").strip()

        if path == "":
            print("Image import cancelled.")
            continue

        if not os.path.isfile(path):
            print("File not found. Please check the path and try again.")
            continue

        try:
            loaded = img.imread(path)
        except Exception as e:
            print(f"Could not open image: {e}")
            continue

        image_path = path
        image_array = loaded
        print(f"Image imported successfully from '{path}'.")
    elif choice == "10":
        confirm = input("All unsaved progress will be erased. Do you wish to exit? (Y/N) ")
        if confirm.lower() == "y":
            print("Exiting...")
            break
        elif confirm.lower() != "n":
            print("Invalid response.")
    else:
        print("Invalid option. Please try again.")
