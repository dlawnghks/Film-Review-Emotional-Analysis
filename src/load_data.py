import os

def load_data(data_dir):
    """
    Loads the IMDb dataset from the given directory.
    
    Args:
        data_dir (str): Path to the dataset directory (e.g., data/aclImdb/train/).
    
    Returns:
        reviews (list): List of review texts.
        labels (list): List of labels (1 for positive, 0 for negative).
    """
    reviews, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        for fname in os.listdir(dir_name):
            with open(os.path.join(dir_name, fname), 'r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return reviews, labels

# Example usage
if __name__ == "__main__":
    train_dir = "data/aclImdb/train"
    test_dir = "data/aclImdb/test"
    
    train_reviews, train_labels = load_data(train_dir)
    test_reviews, test_labels = load_data(test_dir)
    
    print(f"Loaded {len(train_reviews)} training reviews and {len(test_reviews)} test reviews.")
#