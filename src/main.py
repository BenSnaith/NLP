"""
DG3NLP Coursework: Emotional Classification of Tweets
Ben Snaith (230106507)
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# Scikit-learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# ----------------------------------
# GLOBAL STATE
# ----------------------------------

# NOTE: ensure this is the same as in README.md to ensure consistent results
RANDOM_SEED = 230106507
TEST_RATIO = 0.1

EMOTION_LABELS = {
    0: "Anger",
    1: "Fear",
    2: "Joy",
    3: "Love",
    4: "Sadness",
    5: "Surprise",
}

TFIDF_CONFIG = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_iter': 1000,
}

TRANSFORMER_CONFIG = { # applies to both GPT2 and BERT
    'batch_size': 16,
    'learning_rate': 5e-5,
    'num_epochs': 3,
    'max_length': 128,
}

# add mkdirs later

# ----------------------------------
# UTILITY
# ----------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def load_and_split_data(csv_path, test_ratio=1.0):
    print(f"Loading dataset from {csv_path}")

    raw_data = pd.read_csv(csv_path)

    X = raw_data['text'].tolist()
    y = raw_data['label'].tolist()

    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    X, y = list(X), list(y)

    # Split into train/test
    split_idx = int(len(X) * (1 - test_ratio))
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    return X_train, y_train, X_test, y_test

def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(EMOTION_LABELS.values()),
                yticklabels=list(EMOTION_LABELS.values()))
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    # some gui environment do not allow for interactive
    # view of matrices
    plt.close()

def print_classification_metrics(y_true, y_pred, model_name):
    print(f"{'-' * 60}")
    print(f"{model_name} - Report")
    print(f"{'-' * 60}")

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nPerformance Metrics:")
    print(f"\tAccuracy: {accuracy:.4f}")
    print(f"\tMacro F1: {macro_f1:.4f}")
    print(f"\tWeighted F1: {weighted_f1:.4f}")

    print(f"\nPer-class Performance:")
    report = classification_report(
        y_true, y_pred,
        target_names=list(EMOTION_LABELS.values()),
        digits=4,
    )
    print(report)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
    }

# --------------------------------
# MODELS
# --------------------------------

class TFIDFModel:
    def __init__(self, max_features=20000, ngram_range=(1, 2), min_df=2, max_iter=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )

        self.model = LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=max_iter,
            C=1.0,
            class_weight='balanced',
            verbose=0,
        )

    def train(self, X_train, y_train):
        start_time = time()

        print(f"\nVectorising training data with TF-IDF...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        print(f"\nTF-IDF matrix shape: {X_train_vec.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        print(f"\nTraining logistic regression...")
        self.model.fit(X_train_vec, y_train)

        train_time = time() - start_time
        print(f"Training complete in {train_time:.2f} seconds")

        self._show_top_features(n_features=5)

        return self

    def _show_top_features(self, n_features=5):
        print(f"\nTop {n_features} features per class:")
        feature_names = self.vectorizer.get_feature_names_out()

        for emotion_id, emotion_name in EMOTION_LABELS.items():
            if len(self.model.classes_) > 2:
                coef = self.model.coef_[emotion_id]
            else:
                coef = self.model.coef_[0]

            top_indices = np.argsort(coef)[-n_features:][::-1]
            top_features = [feature_names[i] for i in top_indices]

            print(f"\t{emotion_name}: {', '.join(top_features)}")

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

    def predict_proba(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = print_classification_metrics(y_test, y_pred, "TD-IDF Model")
        return y_pred, metrics

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokeniser, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokeniser(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        return{
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=6, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device} (If using CPU this might take a while...)")

        print(f"Loading model {model_name} model...")
        self.tokeniser = BertTokenizer.from_pretrained(model_name)

        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)

        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def train(self, X_train, y_train, X_val, y_val,
              batch_size=16, learning_rate=2e-5, num_epochs=3):
        print(f"{'-' * 60}")
        print("Training Bert Model (Fine-tuned Transformer)")
        print(f"{'-' * 60}")

        train_dataset = EmotionDataset(X_train, y_train, self.tokeniser)
        val_dataset = EmotionDataset(X_val, y_val, self.tokeniser)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimiser = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        best_val_acc = 0
        for epoch in range(num_epochs):
            print(f"{'-' * 60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'-' * 60}")

            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader, desc="Training")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimiser.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss

                loss.backward()
                optimiser.step()

                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}',
                })

            avg_train_loss = total_loss / len(train_loader)
            train_acc = correct / total

            val_acc, val_loss = self._evaluate_loader(val_loader)

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"\tTrain Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"\tVal Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        print(f"Bert Training Complete, Best Validation Accuracy: {best_val_acc:.4f}")
        return self

    def _evaluate_loader(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return accuracy, avg_loss

    def predict(self, X):
        dataset = EmotionDataset(X, [0] * len(X), self.tokeniser)
        dataloader = DataLoader(dataset, batch_size=32)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X):
        dataset = EmotionDataset(X, [0] * len(X), self.tokeniser)
        dataloader = DataLoader(dataset, batch_size=32)

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting probabilities"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = print_classification_metrics(y_test, y_pred, "BERT Model")
        return y_pred, metrics

class EnsembleModel:
    def __init__(self, tfidf_model, transformer_model,
                 baseline_weight=0.3, transformer_weight=0.7):
        self.tfidf = tfidf_model
        self.transformer = transformer_model
        self.baseline_weight = baseline_weight
        self.transformer_weight = transformer_weight

    def predict(self, X):
        print("Making ensemble predictions")

        tfidf_preds = self.tfidf.predict(X)
        transformer_preds = self.transformer.predict(X)

        ensemble_preds = []
        for tf_pred, tr_pred in zip(tfidf_preds, transformer_preds):
            if tf_pred == tr_pred:
                ensemble_preds.append(tf_pred)
            else:
                ensemble_preds.append(tr_pred)

        return np.array(ensemble_preds)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = print_classification_metrics(y_test, y_pred, "Ensemble Model (TF-IDF + BERT)")
        return y_pred, metrics

def main():
    print(f"{'-' * 60}")
    print("DG3NLP Coursework: Emotional Classification of Tweets")
    print("Ben Snaith (230106507)")
    print(f"{'-' * 60}")

    # SET RANDOM SEED
    set_seed(RANDOM_SEED)

    X_train, y_train, X_test, y_test = load_and_split_data(
        '../data/raw/twitter_emotion_data.csv', TEST_RATIO
    )

    # Use some of the training data for validation (transformers)
    val_size = int(len(X_train) * 0.1)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_subset = X_train[:-val_size]
    y_train_subset = y_train[:-val_size]

    results = {}

    # ---------------------------------------
    # TF-IDF + LOGISTIC REGRESSION
    # ---------------------------------------

    print(f"{'-' * 60}")
    print(f"Training TF-IDF...")
    print(f"{'-' * 60}")

    baseline = TFIDFModel(max_features=20000, ngram_range=(1, 2), max_iter=1000)
    baseline.train(X_train, y_train)

    y_test_pred_baseline, test_metrics = baseline.evaluate(X_test, y_test)
    results['baseline'] = test_metrics

    plot_confusion_matrix(
        y_test, y_test_pred_baseline,
        "TF-IDF Model - Confusion Matrix",
        "../results/confusion-matrices/tf_idf_confusion_matrix.png"
    )

    # ---------------------------------------
    # BERT TRANSFORMER
    # ---------------------------------------

    print(f"{'-' * 60}")
    print(f"Training BERT...")
    print(f"NOTE: BERT can take about 10-20 minutes on a modern GPU\nAnd 30-60 minutes on CPU")
    print(f"{'-' * 60}")

    bert_model = BertModel()

    bert_model.train(
        X_train_subset, y_train_subset,
        X_val, y_val,
        batch_size=TRANSFORMER_CONFIG['batch_size'],
        learning_rate=TRANSFORMER_CONFIG['learning_rate'],
        num_epochs=TRANSFORMER_CONFIG['num_epochs'],
    )

    y_test_pred_bert, bert_metrics = bert_model.evaluate(X_test, y_test)
    results['BERT'] = bert_metrics

    plot_confusion_matrix(
        y_test, y_test_pred_bert,
        "BERT Model - Confusion Matrix",
        "../results/confusion-matrices/bert_confusion_matrix.png"
    )

    # ---------------------------------------
    # ENSEMBLE
    # ---------------------------------------

    ensemble = EnsembleModel(baseline, bert_model)
    y_test_pred_ensemble, ensemble_metrics = ensemble.evaluate(X_test, y_test)
    results['ensemble'] = ensemble_metrics

    plot_confusion_matrix(
        y_test, y_test_pred_ensemble,
        "Ensemble Model (TF-IDF + BERT) - Confusion Matrix"
        "../results/confusion-matrices/ensemble_confusion_matrix.png"
    )

    # ---------------------------------------
    # FINAL COMPARISON OF MODELS
    # ---------------------------------------

    print(f"\n{'Model':<20} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name.capitalize():<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['macro_f1']:<12.4f} "
              f"{metrics['weighted_f1']:<12.4f}")

if __name__ == "__main__":
    main()