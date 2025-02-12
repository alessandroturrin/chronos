# https://www.timeseriesclassification.com/dataset.php

from typing import Union, List, Tuple, Any
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from argparse import ArgumentParser

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from aeon.datasets import load_classification

from chronos.chronos import ChronosPipeline, ChronosTokenizer, ChronosModel, ChronosConfig, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM


class TimeSeriesDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class TSConfig:
    def __init__(self, 
                 input_size: int = None,
                 num_classes: int = None,
                 fc1: int = 128,
                 fc2: int = 64,
                 dropout_prob: float = 0.5,
                 negative_slope: float = 0.01) -> None:
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc1 = fc1
        self.fc2 = fc2
        self.dropout_prob = dropout_prob
        self.negative_slope = negative_slope

    def add(self, input_size: int, num_classes: int = None):
        self.input_size = input_size

        if num_classes is not None:
            self.num_classes = num_classes

    def __repr__(self):
        return f'TSConfig(input_size={self.input_size}, num_classes={self.num_classes}, fc1={self.fc1}, fc2={self.fc2}, dropout_prob={self.dropout_prob}, negative_slope={self.negative_slope})'


class TSClassifier(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 num_classes: int, 
                 fc1: int = 128, 
                 fc2: int = 64, 
                 dropout_prob: float = 0.5,
                 negative_slope: float = 0.01) -> None:
        
        super(TSClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, fc1)  
        self.fc2 = nn.Linear(fc1, fc2)          
        self.fc3 = nn.Linear(fc2, num_classes)
        
        self.batch_norm1 = nn.BatchNorm1d(fc1)
        self.batch_norm2 = nn.BatchNorm1d(fc2)
        
        self.relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.batch_norm1(self.fc1(x)) 
        x = self.relu(x)  
        x = self.dropout(x)  
        
        x = self.batch_norm2(self.fc2(x))  
        x = self.relu(x) 
        x = self.dropout(x) 
        
        x = self.fc3(x)
        return self.softmax(x)


class ChronosTSClassifier(ChronosPipeline):

    tokenizer: ChronosTokenizer
    model: ChronosModel
    classifier: TSClassifier

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
        ts_config: TSConfig = None,
        num_epochs: int = 25,
        batch_size: int = 32,
        lr: float = .001,
    ) -> None:
        
        if TSConfig is None:
            self.ts_config = TSConfig()
        else:
            self.ts_config = ts_config

        train_embeddings = self.embed(torch.tensor(context).view(context.shape[0], -1))[0]
        train_embeddings = train_embeddings.view(train_embeddings.shape[0], -1)
        train_dataset = TimeSeriesDataset(train_embeddings, torch.tensor(classes))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        input_size = train_embeddings.shape[1]
        num_classes = len(torch.unique(torch.tensor(classes)))

        self.ts_config.add(input_size, num_classes)
        clf = TSClassifier(input_size, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(clf.parameters(), lr=lr)

        clf.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for _, (embeddings, labels) in enumerate(train_loader):
                optimizer.zero_grad()
            
                outputs = clf(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

            avg_loss = running_loss / len(train_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        self.classifier = SVC(kernel='poly')
        self.classifier.fit(train_embeddings, classes)

    @torch.no_grad()
    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 32
    ) -> torch.Tensor:
        
        
        embeddings, _ = self.embed(context.view(context.shape[0], -1))

        embeddings = embeddings.view(embeddings.shape[0], -1)

        return self.classifier.predict(embeddings)

        test_dataset = TimeSeriesDataset(embeddings, [-1]*embeddings.shape[0])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        predictions = torch.empty(0, dtype=torch.long)
        
        with torch.no_grad():
            for embeddings, _ in test_loader:
                outputs = self.classifier(embeddings)
                _, predicted = torch.max(outputs, 1)
                predictions = torch.cat((predictions, predicted), dim=0)
        
        return predictions

    def save_model(self, output_path: str) -> None:

        torch.save(self.classifier, f'{output_path}/ts_classifier.pth')
    
    def load_model(self, input_path: str) -> None:

        torch.load(f'{input_path}/ts_classifier.pth')

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
        default = 'amazon/chronos-t5-tiny',
        choices = ['amazon/chronos-t5-tiny', 'amazon/chronos-t5-small', 'amazon/chronos-t5-base'],
        type = str
    )

    parser.add_argument(
        '--dataset',
        help = 'select dataset',
        required = False,
        default = 'GunPointAgeSpan',
        choices = ['ACSF1', 'Coffee', 'Earthquakes', 'GunPointAgeSpan'],
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
        '--modality',
        help = 'modality of execution',
        required = False,
        default = 'train',
        choices = ['train', 'test'],
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
        '--lr',
        help='learning rate',
        required = False,
        default = 1e-4,
        type = float
    )

    parser.add_argument(
        '--num_epochs',
        help = 'number of epochs for training',
        required = False,
        default = 1,
        type = int
    )

    parser.add_argument(
        '--batch_size',
        help = 'batch size',
        required = False,
        default = 32,
        type = int
    )

    parser.add_argument(
        '--fc1',
        help = 'size of layer 1',
        required = False,
        default = 256,
        type = int
    )

    parser.add_argument(
        '--fc2',
        help = 'size of layer 2',
        required = False,
        default = 128,
        type = int
    )

    parser.add_argument(
        '--dropout_prob',
        help = 'dropout probability',
        required = False,
        default = 0.5,
        type = float
    )

    parser.add_argument(
        '--negative_slope',
        help = 'negative slope',
        required = False,
        default = 0.01,
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
    modality: str = args.modality
    seed: int = args.seed
    lr: float = args.lr
    num_epochs: int = args.num_epochs
    batch_size: int = args.batch_size
    fc1: int = args.fc1
    fc2: int = args.fc2
    dropout_prob: float = args.dropout_prob
    negative_slope: float = args.negative_slope
    input_path: str = args.input_path
    output_path: str = args.output_path



    # (optional) set seed for reproducibility
    if seed is not None:
        set_seed(seed=seed)

    # set-up for pipeline
    clf = ChronosTSClassifier.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    if modality=='train':
        
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        # TS config for classifier
        ts_config = TSConfig(
            fc1 = fc1,
            fc2 = fc2,
            dropout_prob = dropout_prob,
            negative_slope = negative_slope
        )
        
        X_train, y_train = load_classification(dataset, extract_path=extract_path, split='train', return_metadata=False)
        X_test, y_test = load_classification(dataset, extract_path=extract_path, split='test', return_metadata=False)

        label_encoder = LabelEncoder()
        
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        clf.fit(X_train, y_train, ts_config, num_epochs, batch_size, lr)

        y_pred = clf.predict(torch.tensor(X_test))

        print(accuracy_score(y_test, y_pred))

        clf_2 = SVC()
        clf_2.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_pred_2 = clf_2.predict(X_test.reshape(X_test.shape[0], -1))
        print(accuracy_score(y_test, y_pred_2))

        if output_path is not None:
            clf.save_model(output_path)
    
    elif modality=='test':
        pass