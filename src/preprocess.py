def preprocess_data(reviews, tokenizer, max_length=64):
    """
    Tokenizes and preprocesses the reviews.

    Args:
        reviews (list): List of review texts.
        tokenizer (AutoTokenizer): Tokenizer for preprocessing.
        max_length (int): Maximum token length.

    Returns:
        encodings: Tokenized reviews as tensors.
    """
    return tokenizer(reviews, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
