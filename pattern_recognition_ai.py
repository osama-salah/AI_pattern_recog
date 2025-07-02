import cv2
import base64
import requests
import json
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DetectionResult:
    """Data class to store detection results"""
    objects: List[Dict[str, float]]
    facial_expressions: List[Dict[str, float]]
    confidence_threshold: float = 0.7
    timestamp: str = ""

class GeminiVisionAPI:
    """Handler for Gemini 2.5 Flash Vision API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        
    def encode_image(self, image: np.ndarray) -> str:
        """Encode OpenCV image to base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_image(self, image: np.ndarray) -> Dict:
        """Send image to Gemini for analysis"""
        encoded_image = self.encode_image(image)
        
        prompt = """
        Analyze this image and provide a JSON response with the following structure:
        {
            "objects": [
                {"name": "object_name", "confidence": 0.95, "description": "brief description"}
            ],
            "facial_expressions": [
                {"expression": "expression_name", "confidence": 0.90, "person_id": 1}
            ]
        }
        
        Detect these specific objects:
        - Glass (drinking glass, wine glass, etc.)
        - Bottle (water bottle, wine bottle, etc.)
        - Mobile phone (smartphone, cell phone)
        - Laptop (computer, notebook)
        - Book (textbook, novel, magazine)
        - Cup (coffee cup, tea cup, mug)
        - Pen (ballpoint pen, pencil, marker)
        - Watch (wristwatch, smartwatch)
        - Keys (car keys, house keys)
        - Headphones (earbuds, headset)
        
        Detect these facial expressions:
        - Smile (happy, joyful)
        - Frown (sad, disappointed)
        - Surprised (shocked, amazed)
        - Angry (mad, furious)
        - Neutral (calm, expressionless)
        
        Only include objects and expressions that you can clearly identify with reasonable confidence.
        Provide confidence scores between 0.0 and 1.0.
        """
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": encoded_image
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 1024,
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            json_content = content[json_start:json_end]
            
            return json.loads(json_content)
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return {"objects": [], "facial_expressions": []}

class WebcamCapture:
    """Handle webcam operations"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        
    def initialize_camera(self) -> bool:
        """Initialize the webcam"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if self.cap is None:
            return None
            
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """Release the camera"""
        if self.cap:
            self.cap.release()

class PatternRecognitionAI:
    """Main Pattern Recognition AI class"""
    
    def __init__(self, api_key: str):
        self.gemini_api = GeminiVisionAPI(api_key)
        self.webcam = WebcamCapture()
        self.current_results = DetectionResult([], [])
        self.is_analyzing = False
        self.analysis_count = 0
        
    def start(self):
        """Start the pattern recognition system"""
        if not self.webcam.initialize_camera():
            return
        
        print("🔍 Pattern Recognition AI Started!")
        print("📸 Manual Mode - Press keys to control:")
        print("   SPACE/ENTER - Take photo and analyze")
        print("   'c' - Capture photo without analysis")
        print("   's' - Save current frame")
        print("   'r' - Clear previous results")
        print("   'q' - Quit")
        print("\n⏳ Waiting for your command...")
        
        try:
            while True:
                frame = self.webcam.capture_frame()
                if frame is None:
                    continue
                
                # Display live feed with overlay
                display_frame = self.create_display_frame(frame)
                cv2.imshow('Pattern Recognition AI - Manual Mode', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("👋 Shutting down...")
                    break
                elif key == ord(' ') or key == 13:  # SPACE or ENTER
                    self.capture_and_analyze(frame)
                elif key == ord('c'):
                    self.capture_photo_only(frame)
                elif key == ord('s'):
                    self.save_frame(frame)
                elif key == ord('r'):
                    self.clear_results()
                    
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
        finally:
            self.cleanup()
    
    def capture_and_analyze(self, frame: np.ndarray):
        """Capture photo and analyze it"""
        if self.is_analyzing:
            print("⏳ Analysis in progress, please wait...")
            return
        
        self.analysis_count += 1
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n📸 Capturing and analyzing photo #{self.analysis_count}...")
        print(f"🕒 Time: {timestamp}")
        
        # Save the captured frame
        filename = f"analysis_{self.analysis_count}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"💾 Photo saved as: {filename}")
        
        # Analyze the frame
        self.is_analyzing = True
        self.analyze_frame(frame, timestamp)
        self.is_analyzing = False
        
        print("✅ Analysis complete! Results displayed on screen.")
        print("⏳ Ready for next command...")
    
    def capture_photo_only(self, frame: np.ndarray):
        """Capture photo without analysis"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"captured_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Photo captured and saved as: {filename}")
    
    def analyze_frame(self, frame: np.ndarray, timestamp: str):
        """Analyze frame using Gemini API"""
        print("🤖 Sending to Gemini AI for analysis...")
        
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        if width > 1024:
            scale = 1024 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (new_width, new_height))
        else:
            frame_resized = frame
        
        # Get analysis results
        results = self.gemini_api.analyze_image(frame_resized)
        
        # Update current results
        self.current_results.objects = results.get('objects', [])
        self.current_results.facial_expressions = results.get('facial_expressions', [])
        self.current_results.timestamp = timestamp
        
        # Print results
        self.print_results()
    
    def create_display_frame(self, frame: np.ndarray) -> np.ndarray:
        """Create display frame with overlay information"""
        display = frame.copy()
        height, width = display.shape[:2]
        
        # Create semi-transparent overlay for text
        overlay = display.copy()
        
        # Status indicator
        if self.is_analyzing:
            status_color = (0, 165, 255)  # Orange
            status_text = "🤖 ANALYZING..."
        else:
            status_color = (0, 255, 0)  # Green
            status_text = "📷 READY - Press SPACE to analyze"
        
        # Add status bar at top
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        cv2.putText(display, status_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Add analysis count
        if self.analysis_count > 0:
            count_text = f"Photos analyzed: {self.analysis_count}"
            cv2.putText(display, count_text, (width - 250, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add results overlay if available
        if self.current_results.objects or self.current_results.facial_expressions:
            self.add_results_overlay(display)
        
        # Add control instructions at bottom
        instructions = [
            "SPACE/ENTER: Analyze  |  C: Capture  |  S: Save  |  R: Clear  |  Q: Quit"
        ]
        
        # Semi-transparent bottom bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, height - 40), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        cv2.putText(display, instructions[0], (10, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display
    
    def add_results_overlay(self, frame: np.ndarray):
        """Add results overlay to the frame"""
        height, width = frame.shape[:2]
        y_start = 80
        
        # Semi-transparent background for results
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y_start), (400, min(height - 50, y_start + 300)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        y_offset = y_start + 25
        
        # Add timestamp
        if self.current_results.timestamp:
            cv2.putText(frame, f"Last Analysis: {self.current_results.timestamp}", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_offset += 25
        
        # Add objects
        if self.current_results.objects:
            cv2.putText(frame, "Objects Detected:", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += 20
            
            for obj in self.current_results.objects:
                if obj['confidence'] >= self.current_results.confidence_threshold:
                    text = f"• {obj['name']}: {obj['confidence']:.0%}"
                    cv2.putText(frame, text, (25, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 18
        
        # Add facial expressions
        if self.current_results.facial_expressions:
            y_offset += 10
            cv2.putText(frame, "Facial Expressions:", 
                       (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += 20
            
            for expr in self.current_results.facial_expressions:
                if expr['confidence'] >= self.current_results.confidence_threshold:
                    text = f"• {expr['expression']}: {expr['confidence']:.0%}"
                    cv2.putText(frame, text, (25, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 18
    
    def print_results(self):
        """Print detection results to console"""
        print("\n" + "="*60)
        print(f"🔍 ANALYSIS RESULTS #{self.analysis_count}")
        if self.current_results.timestamp:
            print(f"🕒 Timestamp: {self.current_results.timestamp}")
        print("="*60)
        
        if self.current_results.objects:
            print("\n📦 OBJECTS DETECTED:")
            for i, obj in enumerate(self.current_results.objects, 1):
                if obj['confidence'] >= self.current_results.confidence_threshold:
                    print(f"  {i}. {obj['name']}: {obj['confidence']:.1%} confidence")
                    if 'description' in obj:
                        print(f"    Description: {obj['description']}")
        else:
            print("\nNo objects detected above confidence threshold")
        
        if self.current_results.facial_expressions:
            print("\n📸 FACIAL EXPRESSIONS:")
            for expr in self.current_results.facial_expressions:
                if expr['confidence'] >= self.current_results.confidence_threshold:
                    person_info = f" (Person {expr.get('person_id', 'Unknown')})" if 'person_id' in expr else ""
                    print(f"  • {expr['expression']}: {expr['confidence']:.2%} confidence{person_info}")
        else:
            print("\nNo facial expressions detected above confidence threshold")
        
        print("="*60)
    
    def save_frame(self, frame: np.ndarray):
        """Save current frame with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"captured_frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")
    
    def cleanup(self):
        """Clean up resources"""
        self.webcam.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up successfully")

def main():
    """Main function"""
    # Get API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key in the .env file")
        return
    
    # Create and start the AI system
    ai_system = PatternRecognitionAI(api_key)
    ai_system.start()

if __name__ == "__main__":
    main()