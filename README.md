# Hugging Face fine-tuning example 

This project is an example on how fine-tune an [Hugging Face BERT model](https://huggingface.co/bert-base-uncased) for `text-classification` with the [IMDB Movie Reviews dataset](https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews)

## Dependencies / Environment

- `python=3.7` or higher
- `tensorflow=2.5` or higher
- `transformers=4.12`
- `numpy`
- `pandas`
- `sklearn`
- `matplotlib`
- `seaborn`

## Use it

1. Download the [IMDB Movie Reviews dataset](https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews) on Kaggle and copy the `IMDB Dataset.csv` into the `data/` as `train.csv`.
2. Run the `train.ipynb` notebook.

Results will be stored in the `output` folder.

## To slow on your computer?

If it take too much time to train on your computer or/and the you don't want to manage your dependencies, your can use [Amazon SageMaker](https://aws.amazon.com/en/sagemaker/) managed training infrastructure. For the check the `sagemaker` branch.

`$ git checkout sagemaker`
