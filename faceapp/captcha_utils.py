import random
import string
from PIL import Image, ImageDraw, ImageFont
import io
import base64

def generate_captcha():
    # Generate random text
    length = 6
    characters = string.ascii_uppercase + string.digits
    captcha_text = ''.join(random.choice(characters) for _ in range(length))
    
    # Create image
    width, height = 200, 80
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Use default font or provide path to a font file
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    # Draw text with random positioning and rotation
    for i, char in enumerate(captcha_text):
        x = 20 + i * 30 + random.randint(-5, 5)
        y = 20 + random.randint(-10, 10)
        angle = random.randint(-10, 10)
        
        # Create temporary image for rotation
        char_image = Image.new('RGBA', (40, 40), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_image)
        char_draw.text((10, 5), char, fill='black', font=font)
        
        # Rotate
        rotated_char = char_image.rotate(angle, expand=1)
        image.paste(rotated_char, (x, y), rotated_char)
    
    # Add noise
    for _ in range(100):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        draw.point((x, y), fill=random_color())
    
    # Add lines
    for _ in range(5):
        x1 = random.randint(0, width-1)
        y1 = random.randint(0, height-1)
        x2 = random.randint(0, width-1)
        y2 = random.randint(0, height-1)
        draw.line([(x1, y1), (x2, y2)], fill=random_color(), width=2)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return captcha_text, image_base64

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
