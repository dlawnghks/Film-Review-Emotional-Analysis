from transformers import AutoTokenizer

def preprocess_data(reviews, tokenizer_name="bert-base-uncased", max_length=128):
    """
    Preprocesses the reviews by tokenizing and padding them.
    
    Args:
        reviews (list): List of review texts.
        tokenizer_name (str): Pretrained tokenizer name.
        max_length (int): Maximum sequence length.
    
    Returns:
        encodings: Tokenized and padded reviews.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encodings = tokenizer(reviews, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return encodings
