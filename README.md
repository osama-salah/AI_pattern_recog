# 🔍 AI Pattern Recognition System

A powerful pattern recognition system using Google's Gemini 2.5 Flash AI to detect objects and facial expressions from webcam images through an intuitive Streamlit interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### Object Detection
- **Glass** (drinking glass, wine glass, etc.)
- **Bottle** (water bottle, wine bottle, etc.)
- **Mobile phone** (smartphone, cell phone)
- **Laptop** (computer, notebook)
- **Book** (textbook, novel, magazine)
- **Cup** (coffee cup, tea cup, mug)
- **Pen** (ballpoint pen, pencil, marker)
- **Watch** (wristwatch, smartwatch)
- **Keys** (car keys, house keys)
- **Headphones** (earbuds, headset)

### Facial Expression Recognition
- **Smile** (happy, joyful)
- **Frown** (sad, disappointed)
- **Surprised** (shocked, amazed)
- **Angry** (mad, furious)
- **Neutral** (calm, expressionless)

### Key Capabilities
- 📸 **Real-time webcam capture**
- 🤖 **AI-powered analysis** using Gemini 2.5 Flash
- 🎯 **Confidence scoring** for all detections
- 📊 **Interactive Streamlit interface**
- 💾 **Save results** and analysis history
- 📈 **Cost tracking** for API usage
- 🔧 **Configurable confidence thresholds**

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam access
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/osama-salah/AI_pattern_recog.git
cd AI_pattern_recog
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Get your Gemini API key**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for use in the application

5. **Run the application**
```bash
streamlit run streamlit_app.py
```

## 💻 Usage

1. **Launch the app** and enter your Gemini API key in the sidebar
2. **Adjust settings** like confidence threshold as needed
3. **Capture image** using the webcam capture button
4. **Analyze** the captured image with AI
5. **View results** including detected objects and facial expressions
6. **Save results** for future reference
7. **Review history** of all previous analyses

## 💰 Cost Analysis

The system uses Google's Gemini 2.5 Flash API with the following pricing:
- **Input**: $0.10 per 1M tokens (text/image/video)
- **Output**: $0.40 per 1M tokens

### Expected Costs per Image:
- **Single image analysis**: ~$0.000131 (0.0131¢)
- **500 images/day**: ~$2.00/month
- **2000 images/day**: ~$7.86/month

## 🏗️ Project Structure

```
AI_pattern_recog/
├── streamlit_app.py          # Main Streamlit application
├── pattern_recognition_ai.py # Original OpenCV version
├── config.py                 # Configuration settings
├── cost_calculator.py        # Cost analysis tool
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file (optional):
```bash
GEMINI_API_KEY=your_api_key_here
CONFIDENCE_THRESHOLD=0.7
```

### Customization
- Modify `config.py` to add new object categories or expressions
- Adjust confidence thresholds in the Streamlit sidebar
- Customize the analysis prompt in the `GeminiVisionAPI` class

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for providing the powerful vision API
- **Streamlit** for the excellent web app framework
- **OpenCV** for computer vision capabilities

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/osama-salah/AI_pattern_recog/issues) page
2. Create a new issue with detailed information
3. Contact the maintainer

## 🔮 Future Enhancements

- [ ] Real-time video analysis
- [ ] Batch image processing
- [ ] Custom object training
- [ ] Mobile app version
- [ ] Cloud deployment options
- [ ] Advanced analytics dashboard

---

**Made with ❤️ using Google Gemini AI and Streamlit**
```

## 4. Create a LICENSE file

```text:LICENSE
MIT License

Copyright (c) 2024 AI Pattern Recognition System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 5. Create an environment template

```bash:.env.example
# Gemini API Key (Get from: https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Default settings
CONFIDENCE_THRESHOLD=0.7
CAMERA_INDEX=0
```

## 6. Complete the Streamlit app (fixing the previous incomplete code)

```python:streamlit_app.py
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

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'gemini_api' not in st.session_state:
        st.session_state.gemini_api = None
    
    # Header
    st.title("🔍 AI Pattern Recognition System")
    st.markdown("*Powered by Google Gemini 2.5 Flash*")
    
    # Sidebar
    st.sidebar.title("🔧 Controls & Settings")
    
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
        st.info("🔑 **Get your API key:** Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to create a free API key")
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
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📸 Capture & Analyze", "📊 Latest Results", "📚 History"])
    
    with tab1:
        st.subheader("📸 Image Capture & Analysis