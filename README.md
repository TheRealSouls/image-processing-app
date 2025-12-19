# Image Processing Application with NumPy in Python

In this project, I used the following libraries:
- **NumPy**: one of the main libraries I used. Performs very fast array computations in C rather than Python using vectorised operations. Significantly reduces the time complexity of the program.
- **Matplotlib**: displaying the current image state, without axes through functions like `matplotlib.image` and `matplotlib.pyplot`.
- **OpenCV**: for edge detection and histogram equalisation algorithms, which are much more reliable than ones made from scratch.
- **Tkinter**: used for constructing a GUI (Graphical User Interface) and window.
- **Pillow**: act as a bridge between resulting images and the GUI, so that they can be displayed inside the GUI. Used exclusively for the GUI.
- **os**: used for saving images and uploading them through navigating directories.

```py
import cv2
import helpers # Complicated functions are stored in a separate file.
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import os
```

## 1. What this program does (in plain English)

This program is a **simple image editor**. You give it a picture, and it lets you:

- Turn the picture **black and white (grayscale)**
- **Blur** the picture (make it smoother - Gaussian blur)
- Find the **edges** in the picture (Sobel edge detection)
- **Sharpen** the picture (make details stand out more)
- **Fix bad lighting** (make dark images brighter and clearer - histogram equalisation)
- **Save** the edited picture to a folder
- **Reset** back to the original picture
- **Load a new picture** from your computer

You can use it in **two ways**:
- **Text menu (`main.py`)**: a simple menu in the terminal where you type numbers.
- **GUI window (`gui.py`)**: a window with buttons and sliders.

In this notebook, we will focus on **explaining the code**, especially the **text menu version** in `main.py` and the helper functions in `helpers.py`.


## 2. How `main.py` works (the text menu version)

`main.py` is the version you run in the **terminal**. It:

1. **Loads an image** from a file called `working_image.png`.
2. **Shows a menu** of options (1–10).
3. Waits for you to **type a number**.
4. Based on your choice, it **changes the image** or shows/saves it.
5. It keeps repeating until you choose **10 (Exit)**.

### 2.1. Loading the first image

At the top of `main.py` you have:

```python
image_path = 'working_image.png'
image_array = img.imread(image_path)
```

- **`image_path`** is just the file name of the image.
- **`img.imread(...)`** reads the image file into a **NumPy array** called `image_array`.
- A **NumPy array** is a very fast array. Each pixel represents a colour value in RGB form.

From now on, every time you edit the image, you are really changing this `image_array` variable.


### 2.2. The menu loop

In `main.py` there is a **`while` loop** that keeps asking what you want to do:

```python
choice = "0"

while choice != "10":
    print("Welcome to the image processing toolkit!")
    print("Would you like to:")
    print("1. Display the image")
    # ... other options ...
    print("10. Exit ")

    choice = input("Select your option: ")

    if choice == "1":
        # show image
    elif choice == "2":
        # convert to grayscale
    # ... other choices ...
```

- The loop keeps running **until** you choose **"10"**.
- Inside the loop, `if` / `elif` blocks check your choice and call the right code.
- Each option changes `image_array` or uses it (for example, to display or save it).


### 2.3. What each menu option does (simple overview)

Here is a **very simple summary** of the main options in `main.py`:

- **Option 1 – Display the image**
  - Shows the current `image_array` using `matplotlib.pyplot.imshow`.
  - If the image has colour, it shows it in colour. If it is 2D (grayscale), it uses a gray colour map.

- **Option 2 – Convert to grayscale**
  - If the image has 3 colour channels (Red, Green, Blue), it calls:
    - `helpers.apply_grayscale(image_array)`
  - This turns it into a **single-channel** image (black and white).

- **Option 3 – Blur the image**
  - Asks you for a **blur strength** from 1 to 10.
  - Builds a **Gaussian blur kernel** using `helpers.gaussian_kernel`.
  - Uses `helpers.apply_gaussian_blur` to smooth the image.
  - Larger strength → bigger blur.

- **Option 4 – Edge detection**
  - Makes sure the image is grayscale.
  - Uses **OpenCV Sobel filters** (`cv2.Sobel`) to find edges in x and y directions.
  - Combines them into an **edge magnitude** image and normalizes it.

- **Option 5 – Sharpen the image**
  - Makes sure the values are in a good range (0–255).
  - Calls `helpers.sharpen_image` to apply a sharpening kernel.
  - This makes edges and details more visible.

- **Option 6 – Fix bad lighting (histogram equalisation)**
  - Calls `helpers.histogram_equalisation(image_array)`.
  - For grayscale: spreads out the brightness values so the image is clearer.
  - For colour: equalises the brightness channel in a colour space (LAB) so the result still looks natural.

- **Option 7 – Save Image**
  - Converts the image to `uint8` (0–255 integer pixels) using `helpers.to_uint8`.
  - Asks you for a file name.
  - Makes sure a `saves` folder exists using `os.makedirs`.
  - Uses `cv2.imwrite` to write the image to disk.

- **Option 8 – Reset Image**
  - Reloads the image from `image_path`.
  - This undoes all changes and restores the last loaded file.

- **Option 9 – Import Image**
  - Asks you to type a path to a new image file.
  - Checks that the file exists with `os.path.isfile`.
  - Loads it with `img.imread` and updates `image_path` and `image_array`.

- **Option 10 – Exit**
  - Asks you to confirm that you really want to quit.
  - If you type `Y`, the loop ends and the program stops.


## 3. How the helper functions work (`helpers.py`)

The file `helpers.py` contains **reusable functions** that do the actual image processing. `main.py` just calls these.

We will describe them in **simple language**.

### 3.1. `apply_gaussian_blur(image, kernel)`

```python
def apply_gaussian_blur(image, kernel):
    k = kernel.shape[0] // 2
    blurred = np.zeros_like(image)

    for i in range(k, image.shape[0] - k):
        for j in range(k, image.shape[1] - k):
            region = image[i-k:i+k+1, j-k:j+k+1]
            if image.ndim == 3:
                blurred[i, j] = np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1))
            else:
                blurred[i, j] = np.sum(region * kernel)

    return blurred
```

- Think of the image as a **grid of pixels**.
- For each pixel, we look at a **small square around it** (the `region`).
- We **multiply** this region by the `kernel` (a table of weights) and **add** the results.
- This makes each pixel become a **mix of its neighbours**, which creates a blur.
- For colour images, it does this **for each colour channel separately** (Red, Green, Blue).

### 3.2. `gaussian_kernel(size, sigma)`

```python
def gaussian_kernel(size, sigma):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)
```

- This function **builds the blur kernel** used above.
- `size` controls how big the square is (e.g. 3×3, 5×5, 7×7).
- `sigma` controls **how strong** the blur is.
- It uses a **Gaussian formula** (a bell-shaped curve) so that pixels near the centre count more than far-away pixels.
- At the end, it **divides by the sum** so that the kernel adds up to 1 (this keeps the image from getting too bright or too dark).


### 3.3. `apply_grayscale(image)`

```python
def apply_grayscale(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])
```

- This turns a **colour image** into a **grayscale image**.
- It takes the **Red, Green, Blue** channels and combines them into **one value** per pixel.
- The numbers `[0.299, 0.587, 0.114]` are **weights** that roughly match how our eyes see brightness:
  - Green contributes the most, then red, then blue.
- The result is a 2D array where each number is the **brightness** of that pixel.

### 3.4. `sharpen_image(image)`

```python
def sharpen_image(image):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    k = kernel.shape[0] // 2
    sharpened = np.zeros_like(image)

    for i in range(k, image.shape[0] - k):
        for j in range(k, image.shape[1] - k):
            region = image[i-k:i+k+1, j-k:j+k+1]
            if image.ndim == 3:
                sharpened[i, j] = np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1))
            else:
                sharpened[i, j] = np.sum(region * kernel)

    return np.clip(sharpened, 0, 255)
```

- This function uses a **sharpening kernel**.
- It is similar to the blur function, but the kernel values are different.
- The centre pixel has a **large positive value (5)** and the neighbours have **negative values (-1)**.
- This makes **edges stronger** and details more visible.
- `np.clip(..., 0, 255)` makes sure pixel values stay within a **valid range** for images.


### 3.5. `to_uint8(image)`

```python
def to_uint8(image):
    if image.dtype == np.uint8:
        return image

    img = image.astype(np.float64)

    if img.max() <= 1.0:
        img = img * 255

    return np.clip(img, 0, 255).astype(np.uint8)
```

- Images can be stored in different **number types** and ranges.
- Some libraries expect pixel values to be **integers from 0 to 255** (`uint8`).
- This function:
  - Converts the image to `float64`.
  - If values are between 0 and 1, it **scales them up** to 0–255.
  - Clips everything to stay in 0–255.
  - Converts to `uint8` (an 8-bit unsigned integer type).

This is useful before **saving images** or using some OpenCV functions.

### 3.6. `histogram_equalisation(image)`

```python
def histogram_equalisation(image):
    img = to_uint8(image)

    if img.ndim == 2:
        equalized = cv2.equalizeHist(img)
        return equalized.astype(np.float64)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        bgr_equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb_equalized = cv2.cvtColor(bgr_equalized, cv2.COLOR_BGR2RGB)
        return rgb_equalized.astype(np.float64)
```

- **Goal**: fix images that are too dark or low-contrast.
- For **grayscale images**:
  - It directly applies `cv2.equalizeHist`, which spreads out the brightness values.
- For **colour images**:
  - Converts the image to **BGR** (OpenCV’s colour order).
  - Converts BGR to **LAB** colour space.
  - Only equalises the **L channel** (lightness/brightness).
  - Converts back to BGR, then to **RGB**.
- This keeps the **colours looking natural** while improving brightness and contrast.

