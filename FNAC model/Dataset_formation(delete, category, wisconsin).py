import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import ttk, messagebox
from skimage.segmentation import active_contour
from skimage.filters import sobel
import pandas as pd
from PIL import Image, ImageTk
import os

class CellBoundaryDetector:
    def __init__(self):
        self.boundaries = []  # List to store manual boundaries
        self.manual_boundary_points = []  # Store points of the current manual boundary
        self.refined_boundaries = []  # To keep track of refined boundaries
        self.refined_boundary_ids = []  # Canvas IDs for refined boundaries
        self.drawing = False  # Flag for boundary drawing

        self.image_path = r"D:\anu D\AI in healthcare\Contour detection\MALIGNANT\1211.24\1211.24.b3.jpg"
        self.img = cv2.imread(self.image_path)
        
        # Extract filename for Serial no.
        self.serial_no = os.path.splitext(os.path.basename(self.image_path))[0]

        # Check if the image was successfully loaded
        if self.img is None:
            print(f"Error: Unable to load image at {self.image_path}. Please check the file path or file integrity.")
            self.root = tk.Tk()  # Initialize root even if the image fails to load
            self.root.title("Cell Boundary Detector - Error")
            label = tk.Label(self.root, text="Error: Unable to load image. Check the file path or file integrity.", fg="red")
            label.pack(pady=20)
            self.root.after(5000, self.root.destroy)  # Close the GUI after 5 seconds
            return

        # Preserve the original image dimensions
        self.original_img = self.img.copy()
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.features_list = []  # List to store features of refined cells
        self.zoom_factor = 1.0  # Initial zoom factor
        self.magnification_adjustment_factor = 0.7875  # Adjustment factor for features
        self.setup_gui()

    def setup_gui(self):  
        self.root = tk.Tk()
        self.root.title("Advanced Cell Boundary Detector")

        # Dynamically set the canvas size to match the image dimensions  
        img_height, img_width = self.original_img.shape[:2]

        control_frame = ttk.Frame(self.root)
        control_frame.pack()

        # Add buttons for zoom in, zoom out, delete, and finalize processing
        ttk.Button(control_frame, text="Zoom In", command=self.zoom_in).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Zoom Out", command=self.zoom_out).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Draw Manual Boundary", command=lambda: None).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="Add Snake", command=self.refine_boundary_with_snake).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(control_frame, text="Delete", command=self.delete_last_boundary).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(control_frame, text="Finalize", command=self.finalize_processing).grid(row=0, column=5, padx=5)

        self.canvas = tk.Canvas(self.root, width=img_width, height=img_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind canvas events for drawing
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        self.update_canvas()

    def update_canvas(self):
        # Adjust image size based on zoom factor
        zoomed_img = cv2.resize(
            self.original_img, 
            None, 
            fx=self.zoom_factor, 
            fy=self.zoom_factor, 
            interpolation=cv2.INTER_LINEAR
        )

        # Convert BGR OpenCV image to RGB PIL Image
        self.display_img = cv2.cvtColor(zoomed_img, cv2.COLOR_BGR2RGB)
        self.display_img = Image.fromarray(self.display_img)
        self.photo = ImageTk.PhotoImage(self.display_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(width=self.display_img.width, height=self.display_img.height)
        self.canvas.image = self.photo  # Keep a reference to prevent garbage collection

    def zoom_in(self):
        self.zoom_factor *= 1/0.1938  
        self.update_canvas()

    def zoom_out(self):
        self.zoom_factor /= 1/0.1938  
        self.update_canvas()

    def start_draw(self, event):
        self.drawing = True
        self.manual_boundary_points = [(event.x, event.y)]

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.manual_boundary_points.append((x, y))
            self.canvas.create_line(self.manual_boundary_points[-2], (x, y), fill="green", width=2)

    def end_draw(self, event):
        self.drawing = False
        self.boundaries.append(self.manual_boundary_points)
        self.canvas.create_polygon(self.manual_boundary_points, outline="red", fill="", width=2)

    def refine_boundary_with_snake(self):
        if self.boundaries:
            initial_snake = np.array(self.boundaries[-1], dtype=np.float32)
            refined_snake = active_contour(
                sobel(self.gray_img), 
                initial_snake, 
                alpha=0.006,
                beta=0.25,
                gamma=0.01,
                max_num_iter=2550,
                boundary_condition='periodic'
            )
            boundary_id = self.draw_refined_boundary(refined_snake)
            self.refined_boundaries.append(refined_snake)
            self.refined_boundary_ids.append(boundary_id)
            self.calculate_features(refined_snake)
            print("Boundary refinement with snake model complete.")   

    def draw_refined_boundary(self, refined_boundary):
        refined_points = [(int(x), int(y)) for x, y in refined_boundary]
        return self.canvas.create_polygon(refined_points, outline="blue", fill="", width=2)

    def calculate_features(self, boundary_points):
        contour = np.array(boundary_points, dtype=np.int32)
        area = cv2.contourArea(contour) if len(contour) > 2 else 0
        perimeter = cv2.arcLength(contour, closed=True) if len(contour) > 2 else 0
        radius = math.sqrt(area / math.pi) if area > 0 else 0

        # Calculate smoothness
        center = np.mean(contour, axis=0)  # Approximate center
        distances = np.linalg.norm(contour - center, axis=1)  # Distances from the center
        r_mean = np.mean(distances)
        smoothness = np.mean(np.abs(distances - r_mean) / r_mean ) if r_mean > 0 else 0  

        # Adjust for magnification
        area *= self.magnification_adjustment_factor ** 2
        perimeter *= self.magnification_adjustment_factor
        radius *= self.magnification_adjustment_factor
        smoothness *= self.magnification_adjustment_factor

        self.features_list.append((area, perimeter, radius, smoothness))

    def delete_last_boundary(self):
        if self.refined_boundaries:
            self.refined_boundaries.pop()
            last_refined_id = self.refined_boundary_ids.pop()
            self.canvas.delete(last_refined_id)
            print("Last refined boundary and associated data removed.")
        else:
            print("No boundaries to delete.")

    def finalize_processing(self):
        refined_count = len(self.refined_boundaries)
        if refined_count == 0:
            print("No snake-refined cells to process.")
            return
        
        features = {
            "Serial no.": self.serial_no,
            "area_mean": np.mean([f[0] for f in self.features_list]),
            "area_se": np.std([f[0] for f in self.features_list]) / math.sqrt(refined_count),
            "area_worst": np.max([f[0] for f in self.features_list]),
            "perimeter_mean": np.mean([f[1] for f in self.features_list]),
            "perimeter_se": np.std([f[1] for f in self.features_list]) / math.sqrt(refined_count),
            "perimeter_worst": np.max([f[1] for f in self.features_list]),
            "radius_mean": np.mean([f[2] for f in self.features_list]),
            "radius_se": np.std([f[2] for f in self.features_list]) / math.sqrt(refined_count),
            "radius_worst": np.max([f[2] for f in self.features_list]),
            "smoothness_mean": np.mean([f[3] for f in self.features_list]),
            "smoothness_worst": np.max([f[3] for f in self.features_list]),
            "snake_refined_count": refined_count
            
        }
        print("Finalized Features:", features)
        try:
            output_df = pd.DataFrame(features, index=[0])
            output_path = "Output_Features.xlsx"
            if not os.path.exists(output_path):  # Check if file does not exist
                with pd.ExcelWriter(output_path, mode="w") as writer:
                    output_df.to_excel(writer, index=False)
            else:  # Append to the existing file
                with pd.ExcelWriter(output_path, mode="a", if_sheet_exists="overlay") as writer:
                    output_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
            messagebox.showinfo("Success", f"Features appended to {output_path}")
        except PermissionError:
            print(f"Error: Unable to save {output_path}. Check if the file is open.")
            messagebox.showerror("Error", f"Permission denied for saving to {output_path}. Close the file if it's open.")
        except Exception as e:
            print("An unexpected error occurred:", str(e))
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    detector = CellBoundaryDetector()
    if detector.img is not None:  # Ensure the GUI opens only if the image was successfully loaded
        detector.root.mainloop()
