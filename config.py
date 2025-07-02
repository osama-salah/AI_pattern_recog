"""Configuration settings for Pattern Recognition AI"""

# Object categories to detect
OBJECT_CATEGORIES = [
    "Glass",
    "Bottle", 
    "Mobile phone",
    "Laptop",
    "Book",
    "Cup",
    "Pen",
    "Watch",
    "Keys",
    "Headphones"
]

# Facial expressions to detect
FACIAL_EXPRESSIONS = [
    "Smile",
    "Frown", 
    "Surprised",
    "Angry",
    "Neutral"
]

# API Configuration
GEMINI_CONFIG = {
    "model": "gemini-2.0-flash-exp",
    "temperature": 0.1,
    "max_output_tokens": 1024,
    "timeout": 30
}

# Camera Configuration
CAMERA_CONFIG = {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "analysis_interval": 2.0  # seconds
}

# Detection Configuration
DETECTION_CONFIG = {
    "confidence_threshold": 0.7,
    "max_image_size": 1024  # pixels
}