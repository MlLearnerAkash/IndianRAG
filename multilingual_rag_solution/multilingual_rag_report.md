# Multilingual RAG Pipeline for Indian Languages
## Scientific Report

**Author:** Manus AI  
**Date:** May 24, 2025

## 1. Introduction

This report documents the design, implementation, and evaluation of a multilingual Retrieval-Augmented Generation (RAG) pipeline for Indian languages. The system extends an existing Hindi-focused RAG implementation to support all 12 scheduled Indian languages, enabling users to query in any Indian language while maintaining high-quality context retrieval and answer generation.

## 2. Background and Motivation

India is a linguistically diverse country with 22 officially recognized languages, of which 12 are widely spoken and have significant digital presence. While Hindi is the most widely spoken language, a substantial portion of the population prefers to interact with technology in their regional languages. The original RAG pipeline was designed to work with Hindi documents and queries, limiting its accessibility to non-Hindi speakers.

The motivation for this project was to create a language-agnostic interface that preserves the quality of the original system while expanding its reach to speakers of all major Indian languages. This approach aligns with India's digital inclusion goals and makes information more accessible to a broader audience.

## 3. System Architecture

The multilingual RAG pipeline consists of several interconnected components:

### 3.1 Language Detection

The system uses the `langdetect` library to identify the language of incoming queries. This component is crucial for determining the appropriate translation and response generation paths. The language detection module returns both the detected language code and a confidence score.

### 3.2 Translation Module

A bidirectional translation system powered by Google Translate API (via the `googletrans` library) enables:
- Translation of non-Hindi queries to Hindi for context retrieval
- Translation of prompt templates from Hindi to the query language
- Translation of Hindi answers back to the query language when necessary

The translation module preserves format placeholders in prompt templates using a marker-based approach to ensure proper formatting across languages.

### 3.3 Context Retrieval

The system uses ChromaDB with a multilingual embedding model (`intfloat/multilingual-e5-base`) for semantic search. This embedding model supports multiple languages, allowing for effective cross-lingual retrieval. When a non-Hindi query is received, it is translated to Hindi before retrieval to match the language of the stored documents.

### 3.4 Answer Generation

The answer generation component uses BharatGPT-3B-Indic, a large language model fine-tuned for Indian languages. The system constructs prompts in the user's query language by translating the template and incorporating the retrieved context. The generated answer is then provided in the same language as the query.

### 3.5 Integration Flow

The complete workflow is as follows:

1. Detect the language of the user query
2. Translate the query to Hindi (if not already in Hindi)
3. Retrieve relevant context using the Hindi query
4. Translate the prompt template to the query language
5. Generate an answer using the translated prompt and retrieved context
6. Return the answer in the query language along with the used context

## 4. Implementation Details

### 4.1 Language Support

The system supports all 12 scheduled Indian languages:
- Hindi (hi)
- Bengali (bn)
- Telugu (te)
- Tamil (ta)
- Marathi (mr)
- Urdu (ur)
- Gujarati (gu)
- Kannada (kn)
- Malayalam (ml)
- Punjabi (pa)
- Odia (or)
- Assamese (as)

Additionally, the system handles English queries for broader accessibility.

### 4.2 Prompt Template Translation

A key challenge was preserving format placeholders (`{docs}` and `{query}`) during prompt template translation. We implemented a marker-based approach that:

1. Extracts format placeholders using regular expressions
2. Replaces them with unique markers unlikely to be translated
3. Translates the marked template
4. Restores the original placeholders in the translated text

This ensures that the formatting structure remains intact across all languages.

### 4.3 Memory Efficiency

To maintain memory efficiency as required:

1. We use lightweight libraries for language detection and translation
2. Translation operations are performed only when necessary
3. The system reuses the existing embedding model and LLM without requiring additional models
4. Text processing is done in chunks to avoid memory overload

### 4.4 Error Handling

The system includes robust error handling for:
- Language detection failures (defaults to Hindi)
- Translation errors (returns original text)
- Format preservation issues (uses marker-based approach)
- Query processing failures (provides meaningful error messages)

## 5. Evaluation and Testing

### 5.1 Language Detection Accuracy

Tests confirmed accurate detection of all 12 scheduled Indian languages plus English. The system correctly identified the language of sample queries with high confidence.

### 5.2 Translation Quality

Translation quality was evaluated for:
- Query translation (non-Hindi to Hindi)
- Prompt template translation (Hindi to other languages)
- Answer translation (Hindi to other languages)

Results showed semantically accurate translations with preserved meaning across languages.

### 5.3 Format Preservation

The marker-based approach successfully preserved format placeholders during translation, ensuring that the system could correctly format prompts in all supported languages.

### 5.4 End-to-End Testing

End-to-end tests with queries in multiple languages confirmed that:
- The system correctly detects query language
- Context retrieval works regardless of query language
- Answers are generated in the query language
- Context is properly included in responses

## 6. Limitations and Future Work

### 6.1 Current Limitations

- Translation quality depends on the third-party translation API
- Some nuances may be lost in translation, especially for languages with limited digital resources
- The system assumes that context documents are in Hindi

### 6.2 Future Improvements

- Incorporate specialized translation models for Indian languages (e.g., IndicTrans)
- Support multilingual document storage and retrieval
- Implement language-specific answer extraction patterns
- Add transliteration support for cross-script queries
- Optimize translation caching for frequently used prompts and queries

## 7. Conclusion

The multilingual RAG pipeline successfully extends the original Hindi-focused system to support all 12 scheduled Indian languages. By integrating language detection, translation, and format preservation, the system enables users to query in their preferred language while maintaining high-quality context retrieval and answer generation.

This implementation aligns with the requirements for a concise, contextual, and non-creative answer generation system that works across multiple Indian languages. The modular design allows for future enhancements and optimizations as language technologies continue to evolve.

## 8. References

1. Google Translate API (via googletrans library)
2. langdetect library for language detection
3. intfloat/multilingual-e5-base for multilingual embeddings
4. BharatGPT-3B-Indic for answer generation
5. ChromaDB for vector storage and retrieval
6. FastAPI for API implementation
