import streamlit as st
import cv2
import base64
import requests
import json
import time
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import os
from PIL import Image
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Pattern Recognition AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class DetectionResult:
    """Data class to store detection results"""
    objects: List[Dict[str, float]]
    facial_expressions: List[Dict[str, float]]
    confidence_threshold: float = 0.7
    timestamp: str = ""
    image: Optional[np.ndarray] = None

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
            st.error(f"Error calling Gemini API: {e}")
            return {"objects": [], "facial_expressions": []}

class StreamlitPatternRecognitionAI:
    """Streamlit-based Pattern Recognition AI"""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        if 'gemini_api' not in st.session_state:
            st.session_state.gemini_api = None
    
    def setup_sidebar(self):
        """Setup the sidebar with controls and settings"""
        st.sidebar.title("🔍 Pattern Recognition AI")
        st.sidebar.markdown("*Powered by Gemini 2.5 Flash*")
        
        # API Key input
        api_key = st.sidebar.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if api_key:
            if st.session_state.gemini_api is None:
                st.session_state.gemini_api = GeminiVisionAPI(api_key)
                st.sidebar.success("✅ API Key configured!")
        else:
            st.sidebar.warning("⚠️ Please enter your Gemini API key to continue")
        
        st.sidebar.markdown("---")
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum confidence score to display results"
        )
        
        # Statistics
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Statistics")
        st.sidebar.metric("Total Analyses", st.session_state.analysis_count)
        st.sidebar.metric("Results Stored", len(st.session_state.analysis_results))
        
        # Cost estimation
        if st.session_state.analysis_count > 0:
            estimated_cost = st.session_state.analysis_count * 0.000131
            st.sidebar.metric("Estimated Cost", f"${estimated_cost:.6f}")
        
        return confidence_threshold
    
    def capture_from_webcam(self):
        """Capture image from webcam"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access webcam")
                return None
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            else:
                st.error("❌ Failed to capture image from webcam")
                return None
                
        except Exception as e:
            st.error(f"❌ Webcam error: {e}")
            return None
    
    def analyze_image(self, image: np.ndarray, confidence_threshold: float):
        """Analyze image using Gemini API"""
        if st.session_state.gemini_api is None:
            st.error("❌ Please configure your Gemini API key first")
            return None
        
        with st.spinner("🤖 Analyzing image with Gemini AI..."):
            # Convert RGB to BGR for API
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Resize for faster processing
            height, width = image_bgr.shape[:2]
            if width > 1024:
                scale = 1024 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_bgr = cv2.resize(image_bgr, (new_width, new_height))
            
            # Get analysis results
            results = st.session_state.gemini_api.analyze_image(image_bgr)
            
            # Create result object
            result = DetectionResult(
                objects=results.get('objects', []),
                facial_expressions=results.get('facial_expressions', []),
                confidence_threshold=confidence_threshold,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                image=image
            )
            
            # Update session state
            st.session_state.analysis_count += 1
            st.session_state.analysis_results.append(result)
            
            return result
    
    def display_results(self, result: DetectionResult):
        """Display analysis results"""
        st.subheader(f"🔍 Analysis Results")
        st.caption(f"📅 {result.timestamp}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📦 Objects Detected")
            if result.objects:
                filtered_objects = [
                    obj for obj in result.objects 
                    if obj['confidence'] >= result.confidence_threshold
                ]
                
                if filtered_objects:
                    for i, obj in enumerate(filtered_objects, 1):
                        confidence_color = "green" if obj['confidence'] > 0.8 else "orange" if obj['confidence'] > 0.6 else "red"
                        st.markdown(
                            f"**{i}. {obj['name']}** "
                            f"<span style='color: {confidence_color}'>({obj['confidence']:.1%})</span>",
                            unsafe_allow_html=True
                        )
                        if 'description' in obj and obj['description']:
                            st.caption(f"   {obj['description']}")
                else:
                    st.info("No objects detected above confidence threshold")
            else:
                st.info("No objects detected")
        
        with col2:
            st.markdown("### 😊 Facial Expressions")
            if result.facial_expressions:
                filtered_expressions = [
                    expr for expr in result.facial_expressions 
                    if expr['confidence'] >= result.confidence_threshold
                ]
                
                if filtered_expressions:
                    for i, expr in enumerate(filtered_expressions, 1):
                        confidence_color = "green" if expr['confidence'] > 0.8 else "orange" if expr['confidence'] > 0.6 else "red"
                        person_info = f" (Person {expr.get('person_id', 'Unknown')})" if 'person_id' in expr else ""
                        st.markdown(
                            f"**{i}. {expr['expression']}** "
                            f"<span style='color: {confidence_color}'>({expr['confidence']:.1%})</span>"
                            f"{person_info}",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No facial expressions detected above confidence threshold")
            else:
                st.info("No facial expressions detected")
    
    def display_image_with_results(self, image: np.ndarray, result: DetectionResult):
        """Display image with results overlay"""
        st.subheader("📸 Captured Image")
        
        # Create annotated image
        annotated_image = self.create_annotated_image(image, result)
        
        # Display image
        st.image(annotated_image, caption=f"Analysis performed at {result.timestamp}", use_column_width=True)
    
    def create_annotated_image(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Create annotated image with results overlay"""
        annotated = image.copy()
        height, width = annotated.shape[:2]
        
        # Convert to PIL for text rendering
        pil_image = Image.fromarray(annotated)
        
        # For simplicity, we'll return the original image
        # In a more advanced version, you could add bounding boxes and labels
        return annotated
    
    def save_image_and_results(self, result: DetectionResult):
        """Save image and results"""
        if result.image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save image
            image_filename = f"analysis_{timestamp}.jpg"
            pil_image = Image.fromarray(result.image)
            pil_image.save(image_filename)
            
            # Save results as JSON
            results_data = {
                "timestamp": result.timestamp,
                "confidence_threshold": result.confidence_threshold,
                "objects": result.objects,
                "facial_expressions": result.facial_expressions,
                "image_filename": image_filename
            }
            
            json_filename = f"results_{timestamp}.json"
            with open(json_filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            return image_filename, json_filename
        return None, None
    
    def display_history(self, confidence_threshold: float):
        """Display analysis history"""
        if not st.session_state.analysis_results:
            st.info("📝 No analysis history yet. Capture and analyze some images to see results here!")
            return
        
        st.subheader("📚 Analysis History")
        
        # Display results in reverse chronological order
        for i, result in enumerate(reversed(st.session_state.analysis_results)):
            with st.expander(f"Analysis #{len(st.session_state.analysis_results) - i} - {result.timestamp}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if result.image is not None:
                        st.image(result.image, caption=f"Captured at {result.timestamp}", width=300)
                
                with col2:
                    # Objects
                    st.markdown("**Objects:**")
                    filtered_objects = [obj for obj in result.objects if obj['confidence'] >= confidence_threshold]
                    if filtered_objects:
                        for obj in filtered_objects:
                            st.write(f"• {obj['name']}: {obj['confidence']:.1%}")
                    else:
                        st.write("None detected")
                    
                    # Expressions
                    st.markdown("**Expressions:**")
                    filtered_expressions = [expr for expr in result.facial_expressions if expr['confidence'] >= confidence_threshold]
                    if filtered_expressions:
                        for expr in filtered_expressions:
                            st.write(f"• {expr['expression']}: {expr['confidence']:.1%}")
                    else:
                        st.write("None