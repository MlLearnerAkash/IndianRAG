"""
Test script for multilingual RAG pipeline
Author: Manus AI
Date: May 24, 2025
"""

import sys
import json
from multilingual_utils import (
    detect_language,
    translate_text,
    translate_prompt_template,
    create_multilingual_prompt,
    translate_answer,
    INDIAN_LANGUAGES
)

# Sample prompt template in Hindi
PROMPT_TEMPLATE = '''आप एक बड़े भाषा मॉडल हैं जो दिए गए संदर्भ के आधार पर सवालों का उत्तर देते हैं। नीचे दिए गए निर्देशों का पालन करें:

1. **प्रश्न पढ़ें**:
    - दिए गए सवाल को ध्यान से पढ़ें और समझें।

2. **संदर्भ पढ़ें**:
    - नीचे दिए गए संदर्भ को ध्यानपूर्वक पढ़ें और समझें।

3. **सूचना उत्पन्न करना**:
    - संदर्भ का उपयोग करते हुए, प्रश्न का विस्तृत और स्पष्ट उत्तर तैयार करें।
    - यह सुनिश्चित करें कि उत्तर सीधा, समझने में आसान और तथ्यों पर आधारित हो।

**संदर्भ**:
{docs}

**प्रश्न**:
{query}

उत्तर:'''

# Sample context for testing
SAMPLE_CONTEXT = [
    "प्रधानमंत्री नरेंद्र मोदी ने आज नई दिल्ली में एक नई स्वास्थ्य योजना की घोषणा की।",
    "इस योजना का उद्देश्य ग्रामीण क्षेत्रों में स्वास्थ्य सेवाओं को बेहतर बनाना है।",
    "सरकार इस योजना पर अगले पांच वर्षों में 10,000 करोड़ रुपये खर्च करेगी।"
]

# Test queries in different Indian languages
TEST_QUERIES = {
    "hi": "प्रधानमंत्री ने कौन सी नई योजना की घोषणा की?",  # Hindi
    "bn": "প্রধানমন্ত্রী কোন নতুন প্রকল্পের ঘোষণা করেছেন?",  # Bengali
    "te": "ప్రధాన మంత్రి ఏ కొత్త పథకాన్ని ప్రకటించారు?",  # Telugu
    "ta": "பிரதமர் எந்த புதிய திட்டத்தை அறிவித்தார்?",  # Tamil
    "mr": "पंतप्रधानांनी कोणती नवीन योजना जाहीर केली?",  # Marathi
    "en": "What new scheme did the Prime Minister announce?"  # English (for comparison)
}

def test_language_detection():
    """Test language detection functionality"""
    print("\n=== Testing Language Detection ===")
    for lang_code, query in TEST_QUERIES.items():
        detected_lang, confidence = detect_language(query)
        expected_lang = lang_code
        status = "✓" if detected_lang == expected_lang else "✗"
        print(f"{status} Query: '{query[:30]}...' - Expected: {expected_lang}, Detected: {detected_lang}, Confidence: {confidence:.2f}")

def test_translation():
    """Test translation functionality"""
    print("\n=== Testing Translation ===")
    # Test Hindi to English
    hindi_text = "नमस्ते, आप कैसे हैं?"
    english_translation = translate_text(hindi_text, 'hi', 'en')
    print(f"Hindi to English: '{hindi_text}' -> '{english_translation}'")
    
    # Test English to Hindi
    english_text = "Hello, how are you?"
    hindi_translation = translate_text(english_text, 'en', 'hi')
    print(f"English to Hindi: '{english_text}' -> '{hindi_translation}'")
    
    # Test between Indian languages
    bengali_text = "আপনি কেমন আছেন?"
    tamil_translation = translate_text(bengali_text, 'bn', 'ta')
    print(f"Bengali to Tamil: '{bengali_text}' -> '{tamil_translation}'")

def test_prompt_translation():
    """Test prompt template translation"""
    print("\n=== Testing Prompt Template Translation ===")
    for lang_code in ['hi', 'bn', 'te', 'ta', 'mr']:
        if lang_code == 'hi':
            # Skip Hindi as it's the source language
            continue
        
        translated_prompt = translate_prompt_template(PROMPT_TEMPLATE, lang_code)
        print(f"\nPrompt template translated to {INDIAN_LANGUAGES[lang_code]}:")
        print(f"{translated_prompt[:100]}...")

def test_multilingual_prompt_creation():
    """Test multilingual prompt creation"""
    print("\n=== Testing Multilingual Prompt Creation ===")
    for lang_code, query in TEST_QUERIES.items():
        if lang_code not in ['hi', 'bn', 'ta']:  # Test with a subset for brevity
            continue
            
        formatted_prompt, detected_lang, translated_query = create_multilingual_prompt(
            query, SAMPLE_CONTEXT, PROMPT_TEMPLATE
        )
        
        print(f"\nQuery Language: {INDIAN_LANGUAGES.get(lang_code, 'English')}")
        print(f"Detected Language: {INDIAN_LANGUAGES.get(detected_lang, 'English')}")
        print(f"Original Query: {query}")
        print(f"Translated Query (for retrieval): {translated_query}")
        print(f"Formatted Prompt (first 100 chars): {formatted_prompt[:100]}...")

def main():
    """Main test function"""
    print("Testing Multilingual RAG Pipeline Components")
    
    test_language_detection()
    test_translation()
    test_prompt_translation()
    test_multilingual_prompt_creation()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
