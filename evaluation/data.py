import re
import nltk
import string
import pandas as pd

from typing import List, Tuple, Union
from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing

nltk.download("punkt")


class DataLoader:
    """Prepare and load custom data using OCTIS

    Arguments:
        dataset: The name of the dataset, default options:
                    * trump
                    * 20news

    Usage:

    **Trump** - Unprocessed

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="trump").prepare_docs(save="trump.txt").preprocess_octis(output_folder="trump")
    ```

    **20 Newsgroups** - Unprocessed

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="20news").prepare_docs(save="20news.txt").preprocess_octis(output_folder="20news")
    ```

    **Custom Data**

    Whenever you want to use a custom dataset (list of strings), make sure to use the loader like this:

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs).preprocess_octis(output_folder="my_docs")
    ```
    """

    def __init__(self, dataset: str, docs: List[str], timestamps: List[]):
        self.dataset = dataset
        self.docs = docs
        self.timestamps = timestamps
        self.octis_docs = None
        self.doc_path = None

    def load_octis(self, custom: bool = False) -> Dataset:
        """Get dataset from OCTIS

        Arguments:
            custom: Whether a custom dataset is used or one retrieved from
                    https://github.com/MIND-Lab/OCTIS#available-datasets

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="20news")
        data = dataloader.load_octis(custom=True)
        ```
        """
        data = Dataset()

        if custom:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)

        self.octis_docs = data
        return self.octis_docs


    def preprocess_octis(
        self,
        preprocessor: Preprocessing = None,
        documents_path: str = None,
        output_folder: str = "docs",
    ):
        """Preprocess the data using OCTIS

        Arguments:
            preprocessor: Custom OCTIS preprocessor
            documents_path: Path to the .txt file
            output_folder: Path to where you want to save the preprocessed data

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        dataloader.preprocess_octis(output_folder="my_docs")
        ```

        If you want to use your custom preprocessor:

        ```python
        from evaluation import DataLoader
        from octis.preprocessing.preprocessing import Preprocessing

        preprocessor = Preprocessing(lowercase=False,
                                remove_punctuation=False,
                                punctuation=string.punctuation,
                                remove_numbers=False,
                                lemmatize=False,
                                language='english',
                                split=False,
                                verbose=True,
                                save_original_indexes=True,
                                remove_stopwords_spacy=False)

        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        dataloader.preprocess_octis(preprocessor=preprocessor, output_folder="my_docs")
        ```
        """
        if preprocessor is None:
            preprocessor = Preprocessing(
                lowercase=False,
                remove_punctuation=False,
                punctuation=string.punctuation,
                remove_numbers=False,
                lemmatize=False,
                language="english",
                split=False,
                verbose=True,
                save_original_indexes=True,
                remove_stopwords_spacy=False,
            )
        if not documents_path:
            documents_path = self.doc_path
        dataset = preprocessor.preprocess_dataset(documents_path=documents_path)
        dataset.save(output_folder)
        

    def _save(self, docs: List[str], save: str):
        """Save the documents"""
        with open(save, mode="wt", encoding="utf-8") as myfile:
            myfile.write("\n".join(docs))

        self.doc_path = save
