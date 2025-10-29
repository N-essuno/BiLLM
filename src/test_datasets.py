import numpy as np
from datasets import DatasetDict, load_dataset, Dataset, concatenate_datasets
from datasets.exceptions import DatasetsError
import requests

def unscramble(scrambled_text: str) -> str:
    """Unscramble a string in a bijective manner.

    Args:
        scrambled_text:
            The scrambled string to unscramble.

    Returns:
        The unscrambled string.
    """
    rng = np.random.default_rng(seed=4242)
    permutation = rng.permutation(x=len(scrambled_text))
    inverse_permutation = np.argsort(permutation)
    unscrambled = "".join(scrambled_text[i] for i in inverse_permutation)
    return unscrambled

hf_token = unscramble("HjccJFhIozVymqXDVqTUTXKvYhZMTbfIjMxG_")

def load_raw_data() -> DatasetDict:
    """Load the raw dataset.

    Args:
        dataset_config:
            The configuration for the dataset.
        cache_dir:
            The directory to cache the dataset.

    Returns:
        The dataset.
    """
    dataset = load_dataset(
        path="EuroEval/scala-da",
        token=unscramble("HjccJFhIozVymqXDVqTUTXKvYhZMTbfIjMxG_"),
    )
    assert isinstance(dataset, DatasetDict)  # type: ignore[used-before-def]
    return DatasetDict({key: dataset[key] for key in ["train", "val", "test"] if key in dataset})

# res = load_raw_data()
# print(res)

danske_talemaader = load_dataset("EuroEval/danske-talemaader", token=hf_token)
danske_citizen_tests = load_dataset("EuroEval/danish-citizen-tests-updated", token=hf_token)
scandiqa_da = load_dataset("EuroEval/scandiqa-da-mini", token=hf_token)
angry_tweets = load_dataset("EuroEval/angry-tweets-mini", token=hf_token)

# print(ds)
# print(ds['train'][1])

# print("-"*50)

# print(ds_2)
# print(ds_2['train'][1])

# print("-"*50)

# print(scandiqa_da)
# print(scandiqa_da['train'][1])

# print("-"*50)

print(angry_tweets)
print(angry_tweets['train'][1])

# check number of examples for each class in angry_tweets
print("Angry tweets class distribution:")
labels, counts = np.unique(angry_tweets['train']['label'], return_counts=True)
for label, count in zip(labels, counts):
    print(f"Class {label}: {count} examples")

