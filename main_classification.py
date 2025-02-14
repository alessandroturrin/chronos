# https://www.timeseriesclassification.com/dataset.php

from typing import Union, List, Tuple, Any
from tqdm import tqdm
import random
import joblib

import numpy as np
import torch

from argparse import ArgumentParser

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from aeon.datasets import load_classification
from chronos.chronos import ChronosPipeline, ChronosTokenizer, ChronosModel, ChronosConfig, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from ext.metrics import Metrics, flatten_metrics

class ChronosTSClassifier(ChronosPipeline):

    tokenizer: ChronosTokenizer
    model: ChronosModel

    def __init__(self, tokenizer, model, clf):
        self.tokenizer = tokenizer
        self.model = model
        self.clf = clf

    @torch.no_grad()
    def embed(
        self, context: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Any]:

        context_tensor = self._prepare_and_validate_context(context=context)
        token_ids, attention_mask, tokenizer_state = (
            self.tokenizer.context_input_transform(context_tensor)
        )
        embeddings = self.model.encode(
            input_ids=token_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
        ).cpu()
        return embeddings, tokenizer_state
    
    def fit(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        classes,
        batch_size: int = 32
    ) -> None:
        
        embeddings = []

        for i in range(0, len(context)):
            row = context[i, :, :]
            emb_batch = self.embed(row.clone().detach())[0][0]

            embeddings.append(emb_batch[0, :-1])
        
        self.clf.fit(embeddings, classes)

    @torch.no_grad()
    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 32
    ) -> torch.Tensor:
        
        embeddings = []

        for i in range(0, len(context)):
            row = context[i, :, :]
            emb_batch = self.embed(row.clone().detach())[0][0]

            embeddings.append(emb_batch[0, :-1])
        
        predictions = self.clf.predict(embeddings)

        return predictions
    
    def load(self, path: str):
        self.clf = joblib.load(path + '/chronos_tsad.pkl')
    
    def save(self, path: str):
        joblib.dump(self.clf, path + '/chronos_tsad.pkl')
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        '''
        Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.
        '''

        config = AutoConfig.from_pretrained(*args, **kwargs)

        assert hasattr(config, "chronos_config"), "Not a Chronos config file"

        chronos_config = ChronosConfig(**config.chronos_config)

        if chronos_config.model_type == "seq2seq":
            inner_model = AutoModelForSeq2SeqLM.from_pretrained(*args, **kwargs)
        else:
            assert chronos_config.model_type == "causal"
            inner_model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)

        return cls(
            tokenizer=chronos_config.create_tokenizer(),
            model=ChronosModel(config=chronos_config, model=inner_model),
            clf = RandomForestClassifier()
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__=='__main__':

    parser = ArgumentParser(description='Chronos Time Series Classifier')

    parser.add_argument(
        '--model_name',
        help = 'chronos model',
        required = False,
        default = 'ALL',
        choices = ['amazon/chronos-t5-tiny', 'amazon/chronos-t5-small', 'amazon/chronos-t5-base'],
        type = str
    )

    parser.add_argument(
        '--dataset',
        help = 'select dataset',
        required = False,
        default = 'Earthquakes',
        choices = ['ACSF1', 'Earthquakes', 'Coffee', 'GunPointAgeSpan'],
        type = str
    )

    # METTI A NONE IL DEFAULT
    parser.add_argument(
        '--extract_path',
        help = 'select local path for datasets collections',
        required = False,
        default = '/Users/alessandroturrin/Documents/Code/chr/datasets/aeon',
        type = str
    )

    parser.add_argument(
        '--seed',
        help = 'seed for reproducibility',
        required = False,
        default = None,
        type = int
    )

    parser.add_argument(
        '--input_path',
        help = 'input path to load the model (test only)',
        required = False, 
        default = None,
        type = str
    )

    parser.add_argument(
        '--output_path',
        help = 'output path to save results',
        required = False, 
        default = None,
        type = str
    )


    # parse arguments
    args = parser.parse_args()

    model_name: str = args.model_name
    dataset: str = args.dataset
    extract_path: str = args.extract_path
    seed: int = args.seed
    input_path: str = args.input_path
    output_path: str = args.output_path



    # (optional) set seed for reproducibility
    if seed is not None:
        set_seed(seed=seed)

    if dataset=='ALL':
        datasets = ['ACSF1', 'Earthquakes', 'Coffee', 'GunPointAgeSpan']
    else:
        datasets = [dataset]

    if model_name=='ALL':
        models = ['amazon/chronos-t5-tiny', 'amazon/chronos-t5-small', 'amazon/chronos-t5-base']
    else:
        models = [model_name]

    # init results
    results = dict()

    for model_name in models:
        results[model_name] = Metrics(model_name)


    for _, dataset in tqdm(enumerate(datasets), desc='dataset'):

        for _, model in tqdm(enumerate(models), desc='chronos model'):
            
            # set-up for pipeline
            clf = ChronosTSClassifier.from_pretrained(
                model,
                torch_dtype=torch.float32,
            )
            
            X_train, y_train = load_classification(dataset, extract_path=extract_path, split='train', return_metadata=False)
            X_test, y_test = load_classification(dataset, extract_path=extract_path, split='test', return_metadata=False)

            label_encoder = LabelEncoder()
            
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            clf.fit(torch.tensor(X_train), y_train)

            y_pred = clf.predict(torch.tensor(X_test))

            results[model].add_results(y_test, y_pred)


    final_results = list()

    for model_name in models:
        final_results.append(results[model_name].compute_stats())
    
    flatten_metrics(final_results)