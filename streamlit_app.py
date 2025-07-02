import streamlit as st
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
        
    def encode_image(self, image: Image.Image) -> str:
        """Encode PIL image to base64"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def analyze_image(self, image: Image.Image) -> Dict:
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

def analyze_image_func(image: Image.Image, confidence_threshold: float):
    """Analyze image using Gemini API"""
    if st.session_state.gemini_api is None:
        st.error("❌ Please configure your Gemini API key first")
        return None
    
    with st.spinner("🤖 Analyzing image with Gemini AI..."):
        # Resize for faster processing
        width, height = image.size
        if width > 1024:
            scale = 1024 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Get analysis results
        results = st.session_state.gemini_api.analyze_image(image)
        
        # Create result object
        result = DetectionResult(
            objects=results.get('objects', []),
            facial_expressions=results.get('facial_expressions', []),
            confidence_threshold=confidence_threshold,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image=np.array(image)
        )
        
        # Update session state
        st.session_state.analysis_count += 1
        st.session_state.analysis_results.append(result)
        
        return result

def display_results(result: DetectionResult):
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

def display_history(confidence_threshold: float):
    """Display analysis history"""
    if not st.session_state.analysis_results:
        st.info("📝 No analysis history yet. Upload and analyze some images to see results here!")
        return
    
    st.subheader("📚 Analysis History")
    
    # Display results in reverse chronological order
    for i, result in enumerate(reversed(st.session_state.analysis_results)):
        with st.expander(f"Analysis #{len(st.session_state.analysis_results) - i} - {result.timestamp}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if result.image is not None:
                    st.image(result.image, caption=f"Analyzed at {result.timestamp}", width=300)
            
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
                    st.write("None detected")

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'gemini_api' not in st.session_state:
        st.session_state.gemini_api = None
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    
    # Header
    st.title("🔍 AI Pattern Recognition System")
    st.markdown("*Powered by Google Gemini 2.5 Flash*")
    
    # Sidebar
    st.sidebar.title("🔧 Controls & Settings")
    
    # API Key input - try secrets first, then user input
    api_key = None
    
    # Try to get API key from Streamlit secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.sidebar.success("✅ API Key loaded from secrets!")
    except:
        # If not in secrets, ask user to input
        api_key = st.sidebar.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
    
    if api_key:
        if st.session_state.gemini_api is None:
            st.session_state.gemini_api = GeminiVisionAPI(api_key)
            if "loaded from secrets" not in str(st.sidebar):
                st.sidebar.success("✅ API Key configured!")
    else:
        st.sidebar.warning("⚠️ Please enter your Gemini API key to continue")
        st.info("🔑 **Get your API key:** Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to create a free API key")
        st.info("💡 **For deployment:** Add your API key to Streamlit secrets as `GEMINI_API_KEY`")
        return
    
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
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.analysis_results = []
        st.session_state.analysis_count = 0
        st.session_state.current_result = None
        st.sidebar.success("History cleared!")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📸 Upload & Analyze", "📊 Latest Results", "📚 History"])
    
    with tab1:
        st.subheader("📸 Image Upload & Analysis")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "📁 Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image file for AI analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.markdown("### 📋 Image Info")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"**Format:** {image.format}")
                st.write(f"**File size:** {len(uploaded_file.getvalue())} bytes")
                
                # Analyze button
                if st.button("🤖 Analyze Image", type="primary", use_container_width=True, key="analyze_upload"):
                    result = analyze_image_func(image, confidence_threshold)
                    if result:
                        st.session_state.current_result = result
                        st.success("✅ Analysis completed!")
                        st.rerun()
        
        # Camera input (works on mobile devices)
        st.markdown("---")
        st.subheader("📱 Camera Capture")
        st.info("📱 **Mobile users:** Use the camera button below to take a photo directly!")
        
        camera_image = st.camera_input("📷 Take a photo")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Camera Capture", use_column_width=True)
            
            with col2:
                st.markdown("### 📋 Image Info")
                st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"**Format:** {image.format}")