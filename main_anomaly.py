from typing import Union, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score

from ext.anomaly_datasets import generate_time_series
from chronos.chronos import ChronosPipeline, ChronosTokenizer, ChronosModel, ChronosConfig, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM

class ChronosTSAD(ChronosPipeline):

    tokenizer: ChronosTokenizer
    model: ChronosModel
    classifier: SVC

    def __init__(self, tokenizer, model):
        #super().__init__(inner_model=model.model)
        self.tokenizer = tokenizer
        self.model = model

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
        
        self.clf = SVC()

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
        )

clf = ChronosTSAD.from_pretrained(
    'amazon/chronos-t5-small',
    device_map="cpu",  
    torch_dtype=torch.float16,
)

for _ in range(15):
    data = generate_time_series(10, 200)
    df = pd.DataFrame(data)

    train, test = df.iloc[:800], df.iloc[800:]

    X_train, X_test, y_train, y_test = train['target'], test['target'], train['anomaly'], test['anomaly']

    clf.fit(torch.tensor(np.array(X_train)), torch.tensor(y_train))
    y_pred = clf.predict(torch.tensor(np.array(X_test)))

    score = accuracy_score(y_test, y_pred)
    print(score)

print()

for _ in range(15):
    data = generate_time_series(10, 200)
    df = pd.DataFrame(data)

    train, test = df.iloc[:800], df.iloc[800:]

    X_train, X_test, y_train, y_test = train['target'], test['target'], train['anomaly'], test['anomaly']

    #clf.fit(torch.tensor(np.array(X_train)), torch.tensor(y_train))
    y_pred = clf.predict(torch.tensor(np.array(X_test)))

    score = accuracy_score(y_test, y_pred)
    print(score)

    clf2 = IsolationForest()
    clf2.fit(torch.tensor(np.array(X_test).reshape((-1,1))))
    y_pred = clf2.predict(torch.tensor(np.array(X_test).reshape((-1,1))))
    y_pred[y_pred==1] = 0
    y_pred[y_pred==-1] = 1
    score = accuracy_score(y_test, y_pred)
    print(score)