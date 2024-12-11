Emotional Analysis Project Starts 12/11 Using Machine Learning and Deep Learning

# Sentiment Analysis with IMDb Dataset

This project performs sentiment analysis on movie reviews using the IMDb Large Movie Review Dataset. The goal is to classify movie reviews as positive or negative using deep learning techniques.

---

## Features
- Binary sentiment classification: Positive vs. Negative
- Built with PyTorch and Hugging Face Transformers
- Supports both CPU and GPU for training
- Modular codebase for easy extension

---

## Dataset

This project uses the **IMDb Large Movie Review Dataset v1.0**, provided by Andrew Maas et al. (2011).  
You can download the dataset directly from the official source:  
[IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

### Dataset Structure
- **Train set**: 25,000 labeled reviews (positive/negative)
- **Test set**: 25,000 labeled reviews (positive/negative)
- **Unlabeled set**: 50,000 reviews without labels

After downloading the dataset, extract it into the `data/` directory. The folder structure should look like this:
