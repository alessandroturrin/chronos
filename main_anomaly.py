from typing import Union, List, Tuple, Any
from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import joblib
import torch

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from ext.anomaly_datasets import generate_time_series
from ext.metrics import Metrics, flatten_metrics
from chronos.chronos import ChronosPipeline, ChronosTokenizer, ChronosModel, ChronosConfig, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM

"""
Chronos-based pipeline for time series anomaly detection
"""
class ChronosTSAD(ChronosPipeline):

    tokenizer: ChronosTokenizer
    model: ChronosModel
    classifier: SVC

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

        for i in range(0, context.shape[0], batch_size):
            batch = context.numpy()[i:i + batch_size]
            emb_batch = self.embed(torch.tensor(batch))[0][0]
            embeddings.append(emb_batch[:-1])
        
        embeddings = torch.cat(embeddings, axis=0)

        self.clf.fit(embeddings, classes)

    @torch.no_grad()
    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 32
    ) -> torch.Tensor:
        
        embeddings = []

        for i in range(0, context.shape[0], batch_size):
            batch = context.numpy()[i:i + batch_size]
            emb_batch = self.embed(torch.tensor(batch))[0][0]
            embeddings.append(emb_batch[:-1])
        
        embeddings = torch.cat(embeddings, axis=0)

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
            clf=SVC()
        )


"""
Main
"""
if __name__=='__main__':
    
    parser = ArgumentParser()

    parser.add_argument(
        '--model',
        help = 'select pre-trained chronos model',
        required = False,
        default = 'amazon/chronos-t5-small',
        choices = ['amazon/chronos-t5-tiny', 'amazon/chronos-t5-small', 'amazon/chronos-t5-base', 'amazon/chronos-t5-large'],
        type = str
    )

    parser.add_argument(
        '--execution_mode',
        help = 'set execution mode',
        required = False,
        default = 'zero_shot',
        choices = ['zero_shot', 'in_context'],
        type = str
    )

    parser.add_argument(
        '--n_datasets',
        help = 'number of datasets to be generated',
        required = False,
        default = 5,
        type = int
    )

    parser.add_argument(
        '--max_kernels',
        help = 'number of maximum kernels for data generation',
        required = False,
        default = 10,
        type = int
    )

    parser.add_argument(
        '--max_anomalies',
        help = 'number of maximum anomalies per dataset',
        required = False,
        default = 200,
        type = int
    )

    parser.add_argument(
        '--train_size',
        help = 'index to split dataset',
        required = False,
        default = .8,
        type = float
    )

    parser.add_argument(
        '--train_first',
        help = 'whether to train the model before few shot or not',
        required = False,
        default = True,
        type = bool
    )

    parser.add_argument(
        '--n_datasets_train',
        help = 'number of datasets to train the model',
        required = False,
        default = 30,
        type = int
    )

    parser.add_argument(
        '--model_dir',
        help = 'dir to save/load the model',
        required = False,
        default = '[ChronosTSAD]model',
        type = str
    )

    parser.add_argument(
        '--save',
        help = 'flag to save the model',
        required = False,
        default = True,
        type = bool
    )

    # parse arguments
    args = parser.parse_args()

    chronos_model = args.model
    execution_mode = args.execution_mode
    n_datasets = args.n_datasets
    max_kernels = args.max_kernels
    max_anomalies = args.max_anomalies
    train_size = args.train_size
    train_first = args.train_first
    n_datasets_train = args.n_datasets_train
    model_dir = args.model_dir
    save = args.save


    # initialize anomaly detector
    tsad = ChronosTSAD.from_pretrained(
        chronos_model,
        device_map="cpu",  
        torch_dtype=torch.float16,
    )

    models = {
        'ChronosTSAD': tsad,
        'random_forest': RandomForestClassifier,
        'SVC': SVC,
    }

    # init results
    results = dict()

    for model_name in models.keys():
            results[model_name] = Metrics(model_name)


    # IN-CONTEXT EXECUTION
    if execution_mode=='in_context':

        for _ in tqdm(range(n_datasets), desc='in-context'):
            
            for model_name, model in models.items():
                
                data = generate_time_series(max_kernels, max_anomalies)
                df = pd.DataFrame(data)

                # split train and test
                idx = int(len(df)*train_size)
                train, test = df.iloc[:idx], df.iloc[idx:]
                X_train, X_test, y_train, y_test = train['target'], test['target'], train['anomaly'], test['anomaly']


                if model_name!='ChronosTSAD':
                    clf = model()
                else:
                    clf = ChronosTSAD.from_pretrained(
                        chronos_model,
                        device_map="cpu",  
                        torch_dtype=torch.float16,
                    )


                if model_name=='ChronosTSAD':
                    clf.fit(torch.tensor(np.array(X_train)), torch.tensor(y_train))
                    y_pred = clf.predict(torch.tensor(np.array(X_test)))
                    results[model_name].add_results(y_test, y_pred)
                else:
                    clf.fit(np.array(X_train).reshape(-1,1), y_train)
                    y_pred = clf.predict(np.array(X_test).reshape(-1,1))
                    results[model_name].add_results(y_test, y_pred)
        
        
        final_results = list()

    # ZERO SHOT (only for chronos)
    elif execution_mode=='zero_shot':

        if train_first:
            # train chronos before performing zero-shot
            for _ in tqdm(range(n_datasets_train), desc='ChronosTSAD-training'):
                data = generate_time_series(max_kernels, max_anomalies)
                df = pd.DataFrame(data)

                # split train and test
                idx = int(len(df)*train_size)
                train, test = df.iloc[:idx], df.iloc[idx:]
                X_train, X_test, y_train, y_test = train['target'], test['target'], train['anomaly'], test['anomaly']

                tsad.fit(torch.tensor(np.array(X_train)), torch.tensor(y_train))
        else:
            if not os.path.exists(model_dir):
                raise ValueError('model dir not found')
            
            tsad.load(model_dir)


        for _ in tqdm(range(n_datasets), desc='zero-shot'):
            
            for model_name, model in models.items():
                
                data = generate_time_series(max_kernels, max_anomalies)
                df = pd.DataFrame(data)

                # split train and test
                idx = int(len(df)*train_size)
                train, test = df.iloc[:idx], df.iloc[idx:]
                X_train, X_test, y_train, y_test = train['target'], test['target'], train['anomaly'], test['anomaly']


                if model_name!='ChronosTSAD':
                    clf = model()


                if model_name=='ChronosTSAD': # no fit for chronos
                    y_pred = tsad.predict(torch.tensor(np.array(X_test)))
                    results[model_name].add_results(y_test, y_pred)
                else:
                    clf.fit(np.array(X_train).reshape(-1,1), y_train)
                    y_pred = clf.predict(np.array(X_test).reshape(-1,1))
                    results[model_name].add_results(y_test, y_pred)
        
    else:
        raise ValueError(f'Execution {execution_mode} not found in [in_context, zero_shot]')
    

    
    # check to save the model
    if execution_mode=='zero_shot' and save and train_first:
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        tsad.save(model_dir)
    
    # compute final results
    final_results = list()

    for model_name in models.keys():
        final_results.append(results[model_name].compute_stats())
    
    flatten_metrics(final_results)