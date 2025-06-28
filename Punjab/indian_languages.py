
INDIAN_SCHEDULED_LANGUAGES = [
    "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada",
    "Kashmiri", "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi",
    "Nepali", "Oriya", "Punjabi", "Sanskrit", "Santali", "Sindhi", "Tamil",
    "Telugu", "Urdu"
]

# Mapping from language name to FastText language code and download link
# Only including languages for which FastText models are available and are Indian Scheduled Languages
FASTTEXT_MODEL_MAP = {
    "Assamese": {"code": "as", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.as.300.vec.gz"},
    "Bengali": {"code": "bn", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz"},
    "Gujarati": {"code": "gu", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.gu.300.vec.gz"},
    "Hindi": {"code": "hi", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz"},
    "Kannada": {"code": "kn", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.kn.300.vec.gz"},
    "Malayalam": {"code": "ml", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ml.300.vec.gz"},
    "Marathi": {"code": "mr", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mr.300.vec.gz"},
    "Nepali": {"code": "ne", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ne.300.vec.gz"},
    "Oriya": {"code": "or", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.or.300.vec.gz"},
    "Punjabi": {"code": "pa", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pa.300.vec.gz"}, # Using 'pa' for Punjabi
    "Sanskrit": {"code": "sa", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sa.300.vec.gz"},
    "Sindhi": {"code": "sd", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sd.300.vec.gz"},
    "Tamil": {"code": "ta", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ta.300.vec.gz"},
    "Telugu": {"code": "te", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.te.300.vec.gz"},
    "Urdu": {"code": "ur", "link": "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ur.300.vec.gz"}
}

# Reverse map for language detection to language name
FASTTEXT_CODE_TO_NAME = {v["code"]: k for k, v in FASTTEXT_MODEL_MAP.items()}


