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
    """Handler for Gemini Vision API"""
    
    def __init__(self, api_key: str):
        # Clean up the API key in case the user pasted more than just the key
        if "=" in api_key:
            api_key = api_key.split("=")[-1].strip().strip('"').strip("'")
        self.api_key = api_key
        # Use the latest stable and recommended model for multimodal input
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        
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
            # Defensive check for expected structure
            if 'candidates' in result and result['candidates'] and 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Extract JSON from the response string, which might be wrapped in markdown
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start != -1 and json_end != 0:
                    json_content = content[json_start:json_end]
                    return json.loads(json_content)
            
            st.error("Could not parse a valid JSON response from the API.")
            return {"objects": [], "facial_expressions": []}
            
        except requests.exceptions.RequestException as e:
            st.error(f"Network error calling Gemini API: {e}")
            return {"objects": [], "facial_expressions": []}
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            st.error(f"Error parsing API response: {e}")
            return {"objects": [], "facial_expressions": []}
        except Exception as e:
            st.error(f"An unexpected error occurred with the Gemini API: {e}")
            return {"objects": [], "facial_expressions": []}

def analyze_image_func(image: Image.Image, confidence_threshold: float):
    """Analyze image using Gemini API and update session state."""
    if st.session_state.gemini_api is None:
        st.error("❌ Please configure your Gemini API key first")
        return None
    
    with st.spinner("🤖 Analyzing image with Gemini AI..."):
        # Resize for faster processing and cost-saving
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
        st.session_state.current_result = result
        
        return result

def display_results(result: DetectionResult):
    """Display analysis results for a single detection."""
    st.markdown("---")
    st.subheader(f"📊 Analysis Results")
    st.caption(f"📅 {result.timestamp}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 Objects Detected")
        if result.objects:
            filtered_objects = [
                obj for obj in result.objects 
                if obj.get('confidence', 0) >= result.confidence_threshold
            ]
            
            if filtered_objects:
                for i, obj in enumerate(filtered_objects, 1):
                    confidence = obj.get('confidence', 0)
                    confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                    st.markdown(
                        f"**{i}. {obj.get('name', 'Unknown')}** "
                        f"<span style='color: {confidence_color}'>({confidence:.1%})</span>",
                        unsafe_allow_html=True
                    )
                    if 'description' in obj and obj['description']:
                        st.caption(f"   {obj['description']}")
            else:
                st.info("No objects detected above confidence threshold.")
        else:
            st.info("No objects were detected in the image.")
    
    with col2:
        st.markdown("### 😊 Facial Expressions")
        if result.facial_expressions:
            filtered_expressions = [
                expr for expr in result.facial_expressions 
                if expr.get('confidence', 0) >= result.confidence_threshold
            ]
            
            if filtered_expressions:
                for i, expr in enumerate(filtered_expressions, 1):
                    confidence = expr.get('confidence', 0)
                    confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                    person_info = f" (Person {expr.get('person_id', 'N/A')})" if 'person_id' in expr else ""
                    st.markdown(
                        f"**{i}. {expr.get('expression', 'Unknown')}** "
                        f"<span style='color: {confidence_color}'>({confidence:.1%})</span>"
                        f"{person_info}",
                        unsafe_allow_html=True
                    )
            else:
                st.info("No expressions detected above confidence threshold.")
        else:
            st.info("No facial expressions were detected.")

def display_history(confidence_threshold: float):
    """Display analysis history in expanders."""
    if not st.session_state.analysis_results:
        st.info("📝 No analysis history yet. Upload and analyze some images to see results here!")
        return
    
    st.subheader("📚 Analysis History")
    
    # Display results in reverse chronological order
    for i, result in enumerate(reversed(st.session_state.analysis_results)):
        with st.expander(f"Analysis #{len(st.session_state.analysis_results) - i} - {result.timestamp}"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if result.image is not None:
                    st.image(result.image, caption=f"Analyzed at {result.timestamp}", use_column_width=True)
            
            with col2:
                # Objects
                st.markdown("**Objects:**")
                filtered_objects = [obj for obj in result.objects if obj.get('confidence', 0) >= confidence_threshold]
                if filtered_objects:
                    for obj in filtered_objects:
                        st.write(f"• {obj.get('name', 'N/A')}: {obj.get('confidence', 0):.1%}")
                else:
                    st.write("None detected")
                
                # Expressions
                st.markdown("**Expressions:**")
                filtered_expressions = [expr for expr in result.facial_expressions if expr.get('confidence', 0) >= confidence_threshold]
                if filtered_expressions:
                    for expr in filtered_expressions:
                        st.write(f"• {expr.get('expression', 'N/A')}: {expr.get('confidence', 0):.1%}")
                else:
                    st.write("None detected")

def main():
    """Main Streamlit application entrypoint."""
    
    # Initialize session state variables
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'gemini_api' not in st.session_state:
        st.session_state.gemini_api = None
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'captured_image_data' not in st.session_state:
        st.session_state.captured_image_data = None

    # --- Header ---
    st.title("🔍 AI Pattern Recognition System")
    st.markdown("*Powered by Google Gemini*")
    
    # --- Sidebar ---
    with st.sidebar:
        st.title("🔧 Controls & Settings")
        
        # API Key input - try secrets first, then user input
        api_key = None
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.success("✅ API Key loaded from secrets!")
        except (KeyError, FileNotFoundError):
            api_key = st.text_input(
                "🔑 Gemini API Key",
                type="password",
                help="Enter your Google Gemini API key"
            )
        
        if api_key:
            if st.session_state.gemini_api is None:
                st.session_state.gemini_api = GeminiVisionAPI(api_key)
                if 'secrets' not in str(st.sidebar):
                    st.sidebar.success("✅ API Key configured!")
        else:
            st.warning("⚠️ Please enter your Gemini API key to continue")
            st.info("🔑 **Get your API key:** Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to create a free API key.")
            st.info("💡 **For deployment:** Add your API key to Streamlit secrets as `GEMINI_API_KEY`.")
            return # Stop execution if no API key
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=1.0, value=0.7, step=0.05,
            help="Minimum confidence score to display results"
        )
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Statistics")
        st.metric("Total Analyses", st.session_state.analysis_count)
        st.metric("Results Stored", len(st.session_state.analysis_results))
        
        # Cost estimation
        if st.session_state.analysis_count > 0:
            # Note: This is a rough estimate. Check official Google pricing.
            estimated_cost = st.session_state.analysis_count * 0.000131
            st.metric("Est. Cost (USD)", f"${estimated_cost:.6f}")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_results = []
            st.session_state.analysis_count = 0
            st.session_state.current_result = None
            st.session_state.captured_image_data = None # Clear captured image
            st.success("History cleared!")
            st.rerun()
    
    # --- Main Interface Tabs ---
    tab1, tab2, tab3 = st.tabs(["📸 Upload & Analyze", "📊 Latest Results", "📚 History"])
    
    with tab1:
        st.subheader("📤 Upload from Device")
        uploaded_file = st.file_uploader(
            "📁 Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image file for AI analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            with col2:
                st.markdown("##### 📋 Image Info")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Dimensions:** {image.width} x {image.height}")
                st.write(f"**Format:** {image.format}")
                
                if st.button("🤖 Analyze Uploaded Image", type="primary", use_container_width=True, key="analyze_upload"):
                    result = analyze_image_func(image, confidence_threshold)
                    if result:
                        st.success("✅ Analysis complete!")
                        display_results(result) # Display results directly on this tab
        
        st.markdown("---")
        st.subheader("📷 Capture with Camera")
        camera_image_buffer = st.camera_input(
            "Take a photo using your device's camera",
            help="Works best on mobile devices or laptops with a webcam."
        )

        # If a new photo is taken, save it to session state
        if camera_image_buffer is not None:
            st.session_state.captured_image_data = camera_image_buffer.getvalue()

        # If there's a captured photo in session state, display it and the analyze button
        if st.session_state.captured_image_data is not None:
            image_bytes = st.session_state.captured_image_data
            image = Image.open(io.BytesIO(image_bytes))

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Camera Capture", use_column_width=True)
            with col2:
                st.markdown("##### 📋 Image Info")
                st.write(f"**Dimensions:** {image.width} x {image.height}")
                st.write(f"**Format:** {image.format}")
                
                if st.button("🤖 Analyze Captured Photo", type="primary", use_container_width=True, key="analyze_camera"):
                    result = analyze_image_func(image, confidence_threshold)
                    if result:
                        st.success("✅ Analysis complete!")
                        display_results(result) # Display results directly on this tab


    with tab2:
        st.header("📊 Latest Analysis")
        if st.session_state.current_result:
            result = st.session_state.current_result
            col1, col2 = st.columns([1, 2])
            with col1:
                if result.image is not None:
                    st.image(result.image, caption=f"Analyzed at {result.timestamp}", use_column_width=True)
            with col2:
                # Re-use the display_results function but with a different title
                st.subheader(f"🔍 Analysis Details")
                st.caption(f"📅 {result.timestamp}")
                display_results(result)
        else:
            st.info("📸 Analyze an image from the first tab to see the latest results here.")

    with tab3:
        st.header("📚 Full Analysis History")
        display_history(confidence_threshold)

if __name__ == "__main__":
    main()
