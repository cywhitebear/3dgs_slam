import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm

class SegVisualizer:
    def __init__(self, device="cuda"):
        self.device = device
        model_id = "nvidia/segformer-b5-finetuned-ade-640-640"
        print(f"[Init] Loading {model_id}...")
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
        self.model.eval()
        
        # Get class labels from the model configuration
        self.id2label = self.model.config.id2label
        
        # Generate a distinct random color for each of the 150 ADE20K classes
        np.random.seed(42)
        self.colormap = np.random.randint(0, 255, (150, 3), dtype=np.uint8)

    @torch.no_grad()
    def process_image(self, image_path, output_path):
        # 1. Load and Preprocess
        original_pil = Image.open(image_path).convert("RGB")
        w, h = original_pil.size
        inputs = self.processor(images=original_pil, return_tensors="pt").to(self.device)
        
        # 2. Inference
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # 3. Upscale and Get Predictions
        upscaled_logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        prediction = torch.argmax(upscaled_logits, dim=1).squeeze(0).cpu().numpy()
        
        # 4. Create Color Overlay
        # Map class IDs to their designated colors
        color_mask = self.colormap[prediction]
        
        # Convert original to BGR for OpenCV
        original_bgr = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
        
        # Blend original image and color mask (50% transparency)
        overlay = cv2.addWeighted(original_bgr, 0.5, color_mask, 0.5, 0)
        
        # 5. Add Labels for Detected Classes
        unique_classes = np.unique(prediction)
        label_y_offset = 30
        
        # Sort classes to keep the legend consistent
        for cls_id in sorted(unique_classes):
            label_text = f"{cls_id}: {self.id2label[cls_id]}"
            color = [int(c) for c in self.colormap[cls_id]]
            
            # Draw a small color box and the text label in the top-left
            cv2.rectangle(overlay, (10, label_y_offset - 15), (30, label_y_offset + 5), color, -1)
            cv2.putText(overlay, label_text, (40, label_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, label_text, (40, label_y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            label_y_offset += 25
            if label_y_offset > h - 20: # Prevent legend from going off-screen
                break

        # 6. Save result
        cv2.imwrite(output_path, overlay)

def main():
    # --- DIRECTORY SETTINGS ---
    # Aligned with your Track3 dataset paths
    input_dir = "/media/ee904/DATA1/Track3/image" 
    output_dir = "/media/ee904/DATA1/Track3/classifying"
    # --------------------------

    if not os.path.exists(input_dir):
        print(f"[Error] Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    visualizer = SegVisualizer()

    print(f"[Info] Visualizing {len(img_files)} images...")
    for img_name in tqdm(img_files):
        in_path = os.path.join(input_dir, img_name)
        out_path = os.path.join(output_dir, img_name)
        visualizer.process_image(in_path, out_path)

    print(f"\nDone! Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()