import cv2
import numpy as np
import os
import csv
import random

def generate_wafer_dataset(num_images=1000):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "TrainingData")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "labels.csv")

    max_shift_px = 30.0
    max_rotation_deg = 15.0

    print(f"Generating {num_images} images with broken-cross reference mark in: {output_dir}")

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'shift_x', 'shift_y', 'rotation_deg'])

        for i in range(num_images):
            # 1. Create the Canvas with Synthetic Noise
            img = np.random.randint(220, 256, (256, 256), dtype=np.uint8)

            thickness = 4
            color = (0,)

            # 2. Draw the Static Reference Mark (Broken Outer Cross)
            # The inner cross extends +/- 20 pixels from the center (128).
            # We start these outer arms at +/- 28 pixels to leave a clean 8-pixel gap.
            
            # Top Arm (Vertical line, stopping before the center)
            cv2.line(img, (128, 70), (128, 100), color, thickness)
            
            # Bottom Arm (Vertical line, starting after the center)
            cv2.line(img, (128, 156), (128, 186), color, thickness)
            
            # Left Arm (Horizontal line, stopping before the center)
            cv2.line(img, (70, 128), (100, 128), color, thickness)
            
            # Right Arm (Horizontal line, starting after the center)
            cv2.line(img, (156, 128), (186, 128), color, thickness)

            # 3. Generate Random Transformation Parameters
            shift_x = random.uniform(-max_shift_px, max_shift_px)
            shift_y = random.uniform(-max_shift_px, max_shift_px)
            rotation_deg = random.uniform(-max_rotation_deg, max_rotation_deg)

            # 4. Create the Target Cross on a Separate Blank Layer
            cross_layer = np.full((256, 256), 255, dtype=np.uint8)
            
            # Inner Cross centered at (128, 128) with +/- 20px arms.
            cv2.line(cross_layer, (128, 108), (128, 148), color, thickness) # Vertical
            cv2.line(cross_layer, (108, 128), (148, 128), color, thickness) # Horizontal

            # 5. Apply Affine Transformation (Rotation + Translation)
            M = cv2.getRotationMatrix2D((128, 128), rotation_deg, 1.0)
            M[0, 2] += shift_x
            M[1, 2] += shift_y

            warped_cross = cv2.warpAffine(cross_layer, M, (256, 256), borderValue=255)

            # 6. Composite the Images
            final_img = np.minimum(img, warped_cross)

            # 7. Save Artifacts
            filename = f"wafer_{i:04d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), final_img)

            # Round coordinates to 4 decimal places
            writer.writerow([
                filename, 
                round(shift_x, 4), 
                round(shift_y, 4), 
                round(rotation_deg, 4)
            ])

    print("Generation complete.")

if __name__ == "__main__":
    generate_wafer_dataset(num_images=1000)