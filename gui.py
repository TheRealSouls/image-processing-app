import os
import cv2
import numpy as np
import matplotlib.image as img
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import helpers

# Path to the initial image
IMAGE_PATH = "working_image.png"

while True:
    IMAGE_PATH = input("Enter path to image file: ").strip()
    if not os.path.isfile(IMAGE_PATH):
        print("File not found. Try again.")
        continue
    try:
        image_array = img.imread(IMAGE_PATH)
        break
    except Exception as e:
        print(f"Invalid image file: {e}")


class ImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing Toolkit")
        self.configure(bg="#1e1e1e")

        # State
        self.image_path = IMAGE_PATH
        self.original_image = img.imread(self.image_path)
        self.image_array = self.original_image.copy()
        self.photo_image = None

        # Build UI
        self._build_layout()
        self._update_preview()

    # ---------------- UI LAYOUT -----------------
    def _build_layout(self):
        # Configure grid
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Image preview area
        self.preview_frame = ttk.Frame(self, padding=10)
        self.preview_frame.grid(row=0, column=0, sticky="nsew")
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Label(self.preview_frame, bg="#2b2b2b")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Controls panel styling
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#252526")
        style.configure("TLabel", background="#252526", foreground="#ffffff")
        style.configure("TButton", background="#3c3c3c", foreground="#ffffff", padding=5)
        style.map("TButton", background=[("active", "#007acc")])

        self.controls = ttk.Frame(self, padding=10)
        self.controls.grid(row=0, column=1, sticky="nsew")

        # Sections
        ttk.Label(
            self.controls,
            text="Operations",
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=0, pady=(0, 8), sticky="w")

        ttk.Button(
            self.controls,
            text="Display Original",
            command=self._on_display_original
        ).grid(row=1, column=0, sticky="ew", pady=2)

        ttk.Button(
            self.controls,
            text="Convert to Grayscale",
            command=self._on_grayscale
        ).grid(row=2, column=0, sticky="ew", pady=2)

        # Blur with strength slider
        ttk.Label(
            self.controls,
            text="Blur Strength (1-10)"
        ).grid(row=3, column=0, pady=(10, 2), sticky="w")

        self.blur_scale = ttk.Scale(
            self.controls,
            from_=1,
            to=10,
            orient="horizontal"
        )
        self.blur_scale.set(3)
        self.blur_scale.grid(row=4, column=0, sticky="ew")

        ttk.Button(
            self.controls,
            text="Apply Blur",
            command=self._on_blur
        ).grid(row=5, column=0, sticky="ew", pady=(4, 2))

        ttk.Button(
            self.controls,
            text="Edge Detection",
            command=self._on_edges
        ).grid(row=6, column=0, sticky="ew", pady=2)

        ttk.Button(
            self.controls,
            text="Sharpen Image",
            command=self._on_sharpen
        ).grid(row=7, column=0, sticky="ew", pady=2)

        ttk.Button(
            self.controls,
            text="Fix Bad Lighting",
            command=self._on_hist_equalize
        ).grid(row=8, column=0, sticky="ew", pady=(2, 10))

        # Utility section
        ttk.Label(
            self.controls,
            text="Utility",
            font=("Segoe UI", 12, "bold")
        ).grid(row=9, column=0, pady=(10, 8), sticky="w")

        ttk.Button(
            self.controls,
            text="Open Image…",
            command=self._on_open
        ).grid(row=10, column=0, sticky="ew", pady=2)

        ttk.Button(
            self.controls,
            text="Reset Image",
            command=self._on_reset
        ).grid(row=11, column=0, sticky="ew", pady=2)

        ttk.Button(
            self.controls,
            text="Save Image",
            command=self._on_save
        ).grid(row=12, column=0, sticky="ew", pady=2)

        ttk.Button(
            self.controls,
            text="Exit",
            command=self.destroy
        ).grid(row=13, column=0, sticky="ew", pady=(10, 0))

    # -------------- IMAGE HELPERS ---------------
    def _update_preview(self):
        if self.image_array is None:
            return

        img_uint8 = helpers.to_uint8(self.image_array)

        if img_uint8.ndim == 2:
            pil_img = Image.fromarray(img_uint8, mode="L")
        else:
            pil_img = Image.fromarray(img_uint8)

        max_w, max_h = 800, 600
        pil_img.thumbnail((max_w, max_h), Image.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.configure(image=self.photo_image)

    # -------------- OPERATION HANDLERS ----------
    def _on_display_original(self):
        self.image_array = self.original_image.copy()
        self._update_preview()

    def _on_grayscale(self):
        if self.image_array.ndim == 3:
            self.image_array = helpers.apply_grayscale(self.image_array)
        self._update_preview()

    def _on_blur(self):
        strength = int(round(self.blur_scale.get()))
        strength = max(1, min(10, strength))

        kernel_size = 3 + 2 * (strength - 1)
        sigma = strength

        kernel = helpers.gaussian_kernel(kernel_size, sigma)
        self.image_array = helpers.apply_gaussian_blur(self.image_array, kernel)
        self._update_preview()

    def _on_edges(self):
        if self.image_array.ndim == 3:
            self.image_array = helpers.apply_grayscale(self.image_array)

        img_uint8 = helpers.to_uint8(self.image_array)

        gx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        self.image_array = magnitude.astype(np.uint8)
        self._update_preview()

    def _on_sharpen(self):
        if self.image_array.max() <= 1.0:
            self.image_array = (self.image_array * 255).astype(np.float64)

        self.image_array = helpers.sharpen_image(self.image_array)
        self._update_preview()

    def _on_hist_equalize(self):
        self.image_array = helpers.histogram_equalization(self.image_array)
        self._update_preview()

    def _on_reset(self):
        self.original_image = img.imread(self.image_path)
        self.image_array = self.original_image.copy()
        self._update_preview()

    def _on_open(self):
        filetypes = [
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
            ("All files", "*.*"),
        ]

        filename = filedialog.askopenfilename(
            title="Open image",
            filetypes=filetypes
        )

        if not filename:
            return

        loaded = img.imread(filename)
        self.image_path = filename
        self.original_image = loaded
        self.image_array = loaded.copy()
        self._update_preview()

    def _on_save(self):
        save_img = helpers.to_uint8(self.image_array)

        initial_dir = os.path.join(os.getcwd(), "saves")
        os.makedirs(initial_dir, exist_ok=True)

        filename = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            initialdir=initial_dir
        )

        if not filename:
            return

        if save_img.ndim == 2:
            cv2.imwrite(filename, save_img)
        else:
            cv2.imwrite(
                filename,
                cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            )

        messagebox.showinfo("Saved", f"Image saved to:\n{filename}")


if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()
