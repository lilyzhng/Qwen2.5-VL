import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
import re
import json

# InternVL3 2D Grounding Implementation
class InternVL3Grounder:
    def __init__(self, model_path="OpenGVLab/InternVL3-8B", device="cuda"):
        """
        Initialize InternVL3 for grounding tasks
        
        Args:
            model_path: Path to InternVL3 model (e.g., "OpenGVLab/InternVL3-8B")
            device: Device to run the model on
        """
        self.device = device
        self.model_path = model_path
        
        # Load model and tokenizer
        print(f"Loading {model_path}...")
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=False
        )
        
        # Set up image preprocessing
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        
    def build_transform(self, input_size=448):
        """Build image transformation pipeline"""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio for dynamic preprocessing"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Dynamic image preprocessing for multi-resolution support"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find the closest aspect ratio
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize and split image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
            
        return processed_images

    def load_image(self, image_path, input_size=448, max_num=12):
        """Load and preprocess image for inference"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
            
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values, image.size
    
    def parse_coordinates(self, response):
        """
        Parse bounding box coordinates from model response
        Expected format: [x1, y1, x2, y2] or similar
        """
        # Try to find coordinate patterns in the response
        import re
        
        # Pattern for [x1, y1, x2, y2] format
        pattern1 = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        match1 = re.search(pattern1, response)
        
        if match1:
            coords = [float(x) for x in match1.groups()]
            return coords
        
        # Pattern for <x1><y1><x2><y2> format
        pattern2 = r'<(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)><(\d+(?:\.\d+)?)>'
        match2 = re.search(pattern2, response)
        
        if match2:
            coords = [float(x) for x in match2.groups()]
            return coords
            
        # Pattern for coordinate pairs
        pattern3 = r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)'
        match3 = re.search(pattern3, response)
        
        if match3:
            coords = [float(x) for x in match3.groups()]
            return coords
            
        return None
    
    def ground_object(self, image_path, referring_expression, max_new_tokens=1024):
        """
        Perform 2D grounding on an image given a referring expression
        
        Args:
            image_path: Path to image or PIL Image
            referring_expression: Text description of the object to ground
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            dict with 'response', 'coordinates', and 'image_size'
        """
        # Load and preprocess image
        pixel_values, image_size = self.load_image(image_path, max_num=12)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        
        # Create grounding prompt
        # Based on InternVL documentation format
        question = f"<image>\nPlease provide the bounding box coordinates of the region this sentence describes: <ref>{referring_expression}</ref>"
        
        # Generate response
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        
        # Parse coordinates from response
        coordinates = self.parse_coordinates(response)
        
        return {
            'response': response,
            'coordinates': coordinates,
            'image_size': image_size,
            'referring_expression': referring_expression
        }
    
    def visualize_grounding(self, image_path, result, save_path=None):
        """
        Visualize grounding results by drawing bounding box on image
        
        Args:
            image_path: Path to original image
            result: Result from ground_object method
            save_path: Optional path to save visualization
        """
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            print("PIL ImageDraw required for visualization")
            return None
            
        # Load original image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
            
        draw = ImageDraw.Draw(image)
        
        if result['coordinates']:
            x1, y1, x2, y2 = result['coordinates']
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Add text label
            text = result['referring_expression'][:50] + "..." if len(result['referring_expression']) > 50 else result['referring_expression']
            
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
                
            draw.text((x1, y1-20), text, fill='red', font=font)
        
        if save_path:
            image.save(save_path)
            print(f"Visualization saved to {save_path}")
            
        return image

# Usage example
def main():
    # Initialize grounder
    grounder = InternVL3Grounder("OpenGVLab/InternVL3-8B")
    
    # Example grounding task
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    referring_expression = "the red car in the parking lot"
    
    # Perform grounding
    result = grounder.ground_object(image_path, referring_expression)
    
    print("Grounding Result:")
    print(f"Referring Expression: {result['referring_expression']}")
    print(f"Model Response: {result['response']}")
    print(f"Parsed Coordinates: {result['coordinates']}")
    print(f"Image Size: {result['image_size']}")
    
    # Visualize results
    if result['coordinates']:
        vis_image = grounder.visualize_grounding(image_path, result, "grounding_result.jpg")
        print("Grounding visualization saved!")
    
    return result

if __name__ == "__main__":
    # Example usage
    # result = main()
    
    # Multiple grounding examples
    examples = [
        "the person wearing a red shirt",
        "the black cat on the sofa", 
        "the blue car in the driveway",
        "the book on the table"
    ]
    
    print("InternVL3 2D Grounding Implementation Ready!")
    print("Available models: InternVL3-1B, InternVL3-2B, InternVL3-8B, InternVL3-9B, InternVL3-14B, InternVL3-38B, InternVL3-78B")
    print("Usage: grounder = InternVL3Grounder('OpenGVLab/InternVL3-8B')")
    print("       result = grounder.ground_object(image_path, 'referring expression')")