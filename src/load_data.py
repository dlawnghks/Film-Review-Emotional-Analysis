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
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(data_dir, label_type)
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"Directory not found: {dir_name}")
        
        for fname in os.listdir(dir_name):
            file_path = os.path.join(dir_name, fname)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reviews.append(f.read())
                    labels.append(1 if label_type == 'pos' else 0)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    return reviews, labels
