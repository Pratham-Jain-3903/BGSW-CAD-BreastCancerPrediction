import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from skimage.filters import sobel
from skimage.segmentation import active_contour
import os
import json
from datetime import datetime

class CellBoundaryDetector:
    def __init__(self):
        self.FNAC_FOLDER = r"C:\Users\Pratham Jain\SisterDear\AIHC\Atharva\FNAC"
        self.points_list = []
        self.current_points = []
        self.drawing = False
        self.original_image = None
        self.display_image = None
        self.zoom_factor = 1.0
        self.snake_colors = []
        self.undo_history = []
        self.show_refined = False
        self.refined_snakes = []
        
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Advanced Cell Boundary Detector")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel
        control_panel = ttk.Frame(main_frame)
        control_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # File operations
        file_frame = ttk.LabelFrame(control_panel, text="File Operations", padding=5)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="Load Image", command=self.load_new_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="Save Project", command=self.save_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="Load Project", command=self.load_project).pack(side=tk.LEFT, padx=2)
        
        # Drawing controls
        draw_frame = ttk.LabelFrame(control_panel, text="Drawing Controls", padding=5)
        draw_frame.pack(fill=tk.X, pady=5)
        # ttk.Button(draw_frame, text="Add Snake", command=self.add_snake).pack(side=tk.LEFT, padx=2)
        ttk.Button(draw_frame, text="Undo", command=self.undo_selection).pack(side=tk.LEFT, padx=2)
        ttk.Button(draw_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        
        
        # View controls
        view_frame = ttk.LabelFrame(control_panel, text="View Controls", padding=5)
        view_frame.pack(fill=tk.X, pady=5)
        ttk.Button(view_frame, text="Zoom In", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_frame, text="Zoom Out", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT, padx=2)
        self.show_refined_var = tk.BooleanVar()
        ttk.Checkbutton(view_frame, text="Show Refined", variable=self.show_refined_var, 
                        command=self.toggle_refined_view).pack(side=tk.LEFT, padx=2)
        
        # Processing controls
        process_frame = ttk.LabelFrame(control_panel, text="Processing", padding=5)
        process_frame.pack(fill=tk.X, pady=5)
        ttk.Button(process_frame, text="Refine Boundaries", command=self.refine_boundaries).pack(side=tk.LEFT, padx=2)
        ttk.Button(process_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=2)
        
        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, width=800, height=600)
        self.canvas.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        self.canvas.bind("<MouseWheel>", self.mouse_zoom)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.load_initial_image()
        
    def load_initial_image(self):
        initial_image_path = os.path.join(self.FNAC_FOLDER, "title.png")
        if os.path.exists(initial_image_path):
            self.load_image(initial_image_path)
        else:
            self.status_var.set("No initial image found. Please load an image.")
            messagebox.showinfo("No Image", "Please load an image to begin.")

    def load_new_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.load_image(file_path)

    def add_snake(self):
        if self.current_points and len(self.current_points) >= 2:  # Allow at least 2 points
            self.current_points.append(self.current_points[0])  # Close the polygon
            self.points_list.append(self.current_points)
            self.snake_colors.append(self.get_random_color())
            self.undo_history.append(("add", len(self.points_list) - 1))
            self.current_points = []
            self.update_canvas()
            self.status_var.set(f"Snake added. Total snakes: {len(self.points_list)}")
        else:
            messagebox.showwarning("Warning", "Draw a snake with at least 2 points before adding.")

    
    def load_image(self, image_path):
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            self.display_image = self.original_image.copy()
            self.update_canvas()
            self.status_var.set(f"Loaded image: {os.path.basename(image_path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error loading image")

    def update_canvas(self):
        if self.display_image is None:
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text="No image loaded", fill="red")
            return
        
        # Rest of the update_canvas method remains the same
        height, width = self.display_image.shape[:2]
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)
        resized = cv2.resize(self.display_image, (new_width, new_height))
        
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.draw_all_snakes()

    def start_draw(self, event):
        self.drawing = True
        self.current_points = [(event.x / self.zoom_factor, event.y / self.zoom_factor)]
    
    def draw(self, event):
        if self.drawing:
            x, y = event.x / self.zoom_factor, event.y / self.zoom_factor
            self.current_points.append((x, y))
            self.draw_current_snake()
    
    def end_draw(self, event):
        if self.drawing and len(self.current_points) > 1:  # Changed from > 2 to > 1
            self.current_points.append(self.current_points[0])  # Close the polygon
            self.points_list.append(self.current_points)
            self.snake_colors.append(self.get_random_color())
            self.undo_history.append(("add", len(self.points_list) - 1))
            self.current_points = []
            self.update_canvas()
        self.drawing = False

    def draw_all_snakes(self):
        for points in self.points_list:
            scaled_points = [(x * self.zoom_factor, y * self.zoom_factor) for x, y in points]
            flattened_points = [coord for point in scaled_points for coord in point]
            self.canvas.create_line(flattened_points, fill="green", width=2)

    def draw_current_snake(self):
        # Ensure proper unpacking of x and y coordinates
        scaled_points = [(x * self.zoom_factor, y * self.zoom_factor) for x, y in self.current_points]
        # Flatten the list of tuples for create_line
        flattened_points = [coord for point in scaled_points for coord in point]
        self.canvas.create_line(flattened_points, fill="black", width=2)


    def get_random_color(self):
        return f'#{np.random.randint(0, 256):02x}{np.random.randint(0, 256):02x}{np.random.randint(0, 256):02x}'
    
    def zoom(self, factor):
        self.zoom_factor *= factor
        self.update_canvas()
    
    def mouse_zoom(self, event):
        if event.delta > 0:
            self.zoom(1.1)
        else:
            self.zoom(0.9)
    
    def undo_selection(self):
        if self.undo_history:
            action, index = self.undo_history.pop()
            if action == "add":
                self.points_list.pop(index)
                self.snake_colors.pop(index)
            self.update_canvas()
    
    def clear_all(self):
        self.points_list = []
        self.snake_colors = []
        self.undo_history = []
        self.update_canvas()
    
    # def refine_boundaries(self):
    #     if not self.points_list:
    #         messagebox.showwarning("Warning", "No snakes to refine.")
    #         return

    #     self.refined_snakes = []
    #     edges = sobel(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY))
        
    #     for points in self.points_list:
    #         initial_snake = np.array(points)  # Convert points to a numpy array for processing
            
    #         # Adjust the alpha, beta, gamma parameters to make the snake fit better
    #         refined_snake = active_contour(
    #             edges, initial_snake, 
    #             alpha=0.015,  # Increase to prevent shrinking
    #             beta=0.01,    # Adjust to control rigidity
    #             gamma=0.1,    # Increase to allow better adaptation
    #             max_iterations=2500,  # Ensure enough iterations for convergence
    #             w_edge=0.8,    # Weight for image edge forces
    #             w_line=0       # Zero line weight, so it focuses on edges
    #         )
    #         self.refined_snakes.append(refined_snake)
        
    #     self.show_refined_var.set(True)
    #     self.toggle_refined_view()


    def refine_boundaries(self):
        if not self.points_list:
            messagebox.showwarning("Warning", "No snakes to refine.")
            return

        self.refined_snakes = []
        
        # Apply Gaussian blur to smooth the image and reduce noise
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = sobel(blurred_image)
        
        for points in self.points_list:
            initial_snake = np.array(points)
            refined_snake = active_contour(
                edges,
                initial_snake,
                alpha=0.005, 
                beta=0.01, 
                gamma=0.1, 
                # max_iterations=2500
            )
            self.refined_snakes.append(refined_snake)
        
        self.show_refined_var.set(True)
        self.toggle_refined_view()


    
    def toggle_refined_view(self):
        self.show_refined = self.show_refined_var.get()
        self.update_canvas()
        if self.show_refined:
            self.draw_refined_snakes()
    
    def draw_refined_snakes(self):
        if not self.refined_snakes:
            return
        for snake in self.refined_snakes:
            scaled_snake = [(x * self.zoom_factor, y * self.zoom_factor) for x, y in snake]
            flattened_points = [coord for point in scaled_snake for coord in point]
            self.canvas.create_line(flattened_points, fill='red', width=2)

    
    def export_results(self):
        if not self.points_list:
            messagebox.showwarning("Warning", "No results to export.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_folder = os.path.join(self.FNAC_FOLDER, f"export_{timestamp}")
        os.makedirs(export_folder, exist_ok=True)
        
        # Save original image with overlays
        overlay_image = self.original_image.copy()
        for i, points in enumerate(self.points_list):
            pts = np.array(points, np.int32)
            cv2.polylines(overlay_image, [pts], True, (0, 255, 0), 2)
        
        if self.refined_snakes:
            for snake in self.refined_snakes:
                pts = np.array(snake, np.int32)
                cv2.polylines(overlay_image, [pts], True, (255, 0, 0), 2)
        
        cv2.imwrite(os.path.join(export_folder, "result_overlay.png"), overlay_image)
        
        # Save coordinates
        results = {
            "original_snakes": [points for points in self.points_list],
            # "original_snakes": [points.tolist() for points in self.points_list],
            "refined_snakes": [snake.tolist() for snake in self.refined_snakes] if self.refined_snakes else []
        }
        
        with open(os.path.join(export_folder, "coordinates.json"), 'w') as f:
            json.dump(results, f)
        
        messagebox.showinfo("Export Complete", f"Results exported to {export_folder}")
    
    def save_project(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                filetypes=[("JSON files", "*.json")])
        if file_path:
            project_data = {
                "points_list": self.points_list,
                "snake_colors": self.snake_colors,
                "refined_snakes": [snake.tolist() for snake in self.refined_snakes] if self.refined_snakes else []
            }
            with open(file_path, 'w') as f:
                json.dump(project_data, f)
            self.status_var.set(f"Project saved to {file_path}")
    
    def load_project(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                project_data = json.load(f)
            
            self.points_list = project_data["points_list"]
            self.snake_colors = project_data["snake_colors"]
            self.refined_snakes = [np.array(snake) for snake in project_data["refined_snakes"]]
            self.update_canvas()
            self.status_var.set(f"Project loaded from {file_path}")

if __name__ == "__main__":
    app = CellBoundaryDetector()
    app.root.mainloop()