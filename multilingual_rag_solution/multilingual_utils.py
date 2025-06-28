"""
Multilingual Utilities for RAG Pipeline
Author: Manus AI
Date: May 24, 2025
"""

import os
import json
from langdetect import detect
from googletrans import Translator
from typing import Dict, List, Tuple, Optional, Union
import re

# Define the 12 scheduled Indian languages and their codes
INDIAN_LANGUAGES = {
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'ta': 'Tamil',
    'mr': 'Marathi',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    'or': 'Odia',
    'as': 'Assamese'
}

# Language detection confidence threshold
LANG_DETECT_CONFIDENCE = 0.7

# Initialize translator
translator = Translator()

def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of the input text.
    
    Args:
        text (str): Input text to detect language
        
    Returns:
        Tuple[str, float]: Detected language code and confidence score
    """
    try:
        # Use langdetect to identify the language
        lang_code = detect(text)
        
        # For simplicity, we're using a fixed confidence score since langdetect doesn't provide one
        confidence = LANG_DETECT_CONFIDENCE
        
        # If detected language is not in our supported languages, default to English
        if lang_code not in INDIAN_LANGUAGES and lang_code != 'en':
            print(f"Warning: Detected language '{lang_code}' is not in supported Indian languages.")
            
        return lang_code, confidence
    except Exception as e:
        print(f"Language detection error: {str(e)}")
        # Default to English if detection fails
        return 'en', 0.0

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from source language to target language.
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language code
        target_lang (str): Target language code
        
    Returns:
        str: Translated text
    """
    if source_lang == target_lang:
        return text
        
    try:
        # Use googletrans for translation
        result = translator.translate(text, src=source_lang, dest=target_lang)
        return result.text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        # Return original text if translation fails
        return text

def translate_prompt_template(template: str, query_lang: str) -> str:
    """
    Translate the prompt template to the query language while preserving format placeholders.
    
    Args:
        template (str): Original prompt template (assumed to be in Hindi)
        query_lang (str): Target language code
        
    Returns:
        str: Translated prompt template with preserved format placeholders
    """
    if query_lang == 'hi':  # If query is in Hindi, no need to translate
        return template
    
    # Extract format placeholders to preserve them
    placeholders = re.findall(r'\{([^{}]+)\}', template)
    
    # Replace format placeholders with unique markers that won't be translated
    marked_template = template
    for i, placeholder in enumerate(placeholders):
        marker = f"PLACEHOLDER_{i}_MARKER"
        marked_template = marked_template.replace(f"{{{placeholder}}}", marker)
    
    # Translate the template from Hindi to the query language
    translated_marked_template = translate_text(marked_template, 'hi', query_lang)
    
    # Restore format placeholders
    translated_template = translated_marked_template
    for i, placeholder in enumerate(placeholders):
        marker = f"PLACEHOLDER_{i}_MARKER"
        translated_template = translated_template.replace(marker, f"{{{placeholder}}}")
    
    return translated_template

def create_multilingual_prompt(query: str, context_docs: List[str], prompt_template: str) -> Tuple[str, str, str]:
    """
    Create a multilingual prompt based on the query language.
    
    Args:
        query (str): User query
        context_docs (List[str]): Retrieved context documents
        prompt_template (str): Original prompt template (in Hindi)
        
    Returns:
        Tuple[str, str, str]: Formatted prompt, detected language code, translated query
    """
    # Detect the language of the query
    query_lang, confidence = detect_language(query)
    
    # If the detected language is not supported, default to Hindi
    if query_lang not in INDIAN_LANGUAGES and query_lang != 'en':
        query_lang = 'hi'
    
    # Translate the query to Hindi for retrieval if it's not already in Hindi
    translated_query = query
    if query_lang != 'hi':
        translated_query = translate_text(query, query_lang, 'hi')
    
    # Translate the prompt template to the query language
    translated_template = translate_prompt_template(prompt_template, query_lang)
    
    # Format the prompt with the translated template, original context, and original query
    formatted_prompt = translated_template.format(
        docs="\n".join(context_docs),
        query=query
    )
    
    return formatted_prompt, query_lang, translated_query

def translate_answer(answer: str, source_lang: str, target_lang: str) -> str:
    """
    Translate the generated answer from source language to target language.
    
    Args:
        answer (str): Generated answer
        source_lang (str): Source language code
        target_lang (str): Target language code
        
    Returns:
        str: Translated answer
    """
    if source_lang == target_lang:
        return answer
        
    return translate_text(answer, source_lang, target_lang)
