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

# Sentiment Analysis with Large Movie Review Dataset

## Overview
This project is a sentiment analysis implementation using the [Large Movie Review Dataset v1.0](https://ai.stanford.edu/~amaas/data/sentiment/).  
The dataset provides 50,000 labeled movie reviews (25,000 for training and 25,000 for testing), as introduced in the following paper:

> Maas, Andrew L., et al. (2011). "Learning Word Vectors for Sentiment Analysis."  
> Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies.  
> [http://www.aclweb.org/anthology/P11-1015](http://www.aclweb.org/anthology/P11-1015)

## Citation
If you use this dataset, please cite:
```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
