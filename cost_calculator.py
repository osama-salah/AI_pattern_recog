"""
Cost Calculator for Pattern Recognition AI using Gemini 2.5 Flash
"""

class GeminiCostCalculator:
    def __init__(self):
        # Pricing per 1M tokens (as of latest pricing)
        self.input_price_per_1m_tokens = {
            'text': 0.10,
            'image': 0.10,
            'video': 0.10,
            'audio': 0.70
        }
        self.output_price_per_1m_tokens = 0.40
        
    def estimate_image_tokens(self, image_resolution="1280x720"):
        """
        Estimate token count for image input
        Gemini typically uses ~258 tokens per image regardless of size
        (images are processed at fixed resolution internally)
        """
        # Based on Gemini documentation and testing
        base_image_tokens = 258
        
        # Resolution factor (higher res might use slightly more tokens)
        resolution_factors = {
            "640x480": 1.0,
            "1280x720": 1.1,
            "1920x1080": 1.2,
            "1024x1024": 1.15
        }
        
        factor = resolution_factors.get(image_resolution, 1.1)
        return int(base_image_tokens * factor)
    
    def estimate_text_prompt_tokens(self):
        """
        Estimate tokens for our detailed prompt
        """
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
        
        # Rough estimation: ~1 token per 4 characters for English text
        char_count = len(prompt)
        estimated_tokens = char_count // 4
        return estimated_tokens
    
    def estimate_output_tokens(self):
        """
        Estimate tokens for typical JSON response
        """
        typical_response = """
        {
            "objects": [
                {"name": "Mobile phone", "confidence": 0.92, "description": "Smartphone held in hand"},
                {"name": "Watch", "confidence": 0.85, "description": "Wristwatch on left wrist"},
                {"name": "Cup", "confidence": 0.78, "description": "Coffee mug on desk"}
            ],
            "facial_expressions": [
                {"expression": "Smile", "confidence": 0.88, "person_id": 1},
                {"expression": "Neutral", "confidence": 0.72, "person_id": 2}
            ]
        }
        """
        
        # Estimate tokens for typical response
        char_count = len(typical_response)
        estimated_tokens = char_count // 4
        
        # Add buffer for variation in response length
        return int(estimated_tokens * 1.5)
    
    def calculate_single_image_cost(self, image_resolution="1280x720"):
        """
        Calculate total cost for processing a single image
        """
        # Input tokens
        image_tokens = self.estimate_image_tokens(image_resolution)
        text_prompt_tokens = self.estimate_text_prompt_tokens()
        total_input_tokens = image_tokens + text_prompt_tokens
        
        # Output tokens
        output_tokens = self.estimate_output_tokens()
        
        # Calculate costs
        input_cost = (total_input_tokens / 1_000_000) * self.input_price_per_1m_tokens['image']
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m_tokens
        total_cost = input_cost + output_cost
        
        return {
            'input_tokens': {
                'image_tokens': image_tokens,
                'text_tokens': text_prompt_tokens,
                'total': total_input_tokens
            },
            'output_tokens': output_tokens,
            'costs': {
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost
            },
            'cost_breakdown': {
                'input_cost_cents': input_cost * 100,
                'output_cost_cents': output_cost * 100,
                'total_cost_cents': total_cost * 100
            }
        }
    
    def calculate_usage_scenarios(self):
        """
        Calculate costs for different usage scenarios
        """
        scenarios = {
            'single_image': 1,
            'per_minute_continuous': 30,  # 30 images per minute (every 2 seconds)
            'per_hour_continuous': 1800,  # 30 images/min * 60 min
            'daily_moderate': 500,        # 500 images per day
            'daily_heavy': 2000          # 2000 images per day
        }
        
        single_cost = self.calculate_single_image_cost()['costs']['total_cost']
        
        results = {}
        for scenario, image_count in scenarios.items():
            total_cost = single_cost * image_count
            results[scenario] = {
                'image_count': image_count,
                'total_cost_usd': total_cost,
                'total_cost_cents': total_cost * 100
            }
        
        return results

def main():
    calculator = GeminiCostCalculator()
    
    print("=" * 60)
    print("GEMINI 2.5 FLASH - PATTERN RECOGNITION AI COST ANALYSIS")
    print("=" * 60)
    
    # Single image analysis
    single_image_cost = calculator.calculate_single_image_cost()
    
    print("\n📊 SINGLE IMAGE PROCESSING COST:")
    print("-" * 40)
    print(f"Input Tokens:")
    print(f"  • Image tokens: {single_image_cost['input_tokens']['image_tokens']:,}")
    print(f"  • Text prompt tokens: {single_image_cost['input_tokens']['text_tokens']:,}")
    print(f"  • Total input tokens: {single_image_cost['input_tokens']['total']:,}")
    
    print(f"\nOutput Tokens: {single_image_cost['output_tokens']:,}")
    
    print(f"\n💰 COST BREAKDOWN:")
    print(f"  • Input cost: ${single_image_cost['costs']['input_cost']:.6f} ({single_image_cost['cost_breakdown']['input_cost_cents']:.4f}¢)")
    print(f"  • Output cost: ${single_image_cost['costs']['output_cost']:.6f} ({single_image_cost['cost_breakdown']['output_cost_cents']:.4f}¢)")
    print(f"  • TOTAL COST: ${single_image_cost['costs']['total_cost']:.6f} ({single_image_cost['cost_breakdown']['total_cost_cents']:.4f}¢)")
    
    # Usage scenarios
    scenarios = calculator.calculate_usage_scenarios()
    
    print(f"\n📈 USAGE SCENARIOS:")
    print("-" * 40)
    
    for scenario, data in scenarios.items():
        scenario_name = scenario.replace('_', ' ').title()
        if data['total_cost_usd'] < 0.01:
            cost_display = f"{data['total_cost_cents']:.4f}¢"
        else:
            cost_display = f"${data['total_cost_usd']:.4f}"
        
        print(f"  • {scenario_name}: {data['image_count']:,} images = {cost_display}")
    
    # Cost optimization tips
    print(f"\n💡 COST OPTIMIZATION TIPS:")
    print("-" * 40)
    print("  • Increase analysis interval (currently 2 seconds)")
    print("  • Reduce image resolution before sending to API")
    print("  • Implement local pre-filtering to reduce API calls")
    print("  • Use confidence thresholds to avoid re-analyzing similar scenes")
    print("  • Batch multiple frames for analysis when possible")
    
    # Monthly estimates
    print(f"\n📅 MONTHLY COST ESTIMATES:")
    print("-" * 40)
    daily_moderate = scenarios['daily_moderate']['total_cost_usd'] * 30
    daily_heavy = scenarios['daily_heavy']['total_cost_usd'] * 30
    continuous_hour = scenarios['per_hour_continuous']['total_cost_usd']
    
    print(f"  • Moderate use (500 images/day): ${daily_moderate:.2f}/month")
    print(f"  • Heavy use (2000 images/day): ${daily_heavy:.2f}/month")
    print(f"  • 1 hour continuous daily: ${continuous_hour * 30:.2f}/month")
    
    print(f"\n⚠️  NOTE: Prices are based on current Gemini 2.5 Flash pricing")
    print(f"   and may vary. Always check latest pricing from Google AI.")

if __name__ == "__main__":
    main()