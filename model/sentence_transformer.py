import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class SentenceTransformer(nn.Module):
    """
    Wrapper over a pre-trained BERT model that takes in sentences and outputs sentence embeddings.
    """
    def __init__(self, pretrained_model_name='bert-base-cased'):
        super(SentenceTransformer, self).__init__()

        self.transformer = BertModel.from_pretrained(pretrained_model_name)
        # self.fc = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state
        # print(last_hidden_state.shape)  # Shape - (batch_size, num_tokens, hidden_dim)
        out = last_hidden_state.mean(dim=1)  # Take the mean of the token embeddings to use as a sentence embedding
        # print(out.shape)  # Shape - (batch_size, hidden_dim)

        # out = self.fc(out)
        # print(out.shape)  # Shape - (batch_size, embed_dim)
        return out


class MultiTaskModel(SentenceTransformer):
    """
    Extend the SentenceTransformer class to support multitask learning (Sentence Classification and Sentiment Analysis)
    """
    def __init__(self, num_classes, num_sentiments, hidden_dim=512, pretrained_model_name='bert-base-cased'):
        super(MultiTaskModel, self).__init__(pretrained_model_name)

        bert_last_hidden_dim = 768

        # Task A - Sentence Classification
        self.classification_head = nn.ModuleList([
            nn.Linear(bert_last_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        ])

        # Task B - Sentiment Analysis
        self.sentiment_head = nn.ModuleList([
            nn.Linear(bert_last_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_sentiments)
        ])

    def forward(self, input_ids, attention_mask):
        """
        Sends sentences through the base SentenceTransformer model and passes the sentence embeddings through the two task heads.
        Returns the outputs of the two task heads which contain class/label probabilities.
        """
        # Call the forward method of parent class (SentenceTransformer) to get the sentence embeddings
        out = super().forward(input_ids, attention_mask)

        # Pass the sentence embeddings through sentence classifier head
        classification_out = out
        for module in self.classification_head:
            classification_out = module(classification_out)
        classification_probs = F.softmax(classification_out, dim=1)  # Take softmax along the logits dimension

        # Pass the sentence embeddings through sentiment analysis head
        sentiment_out = out
        for module in self.sentiment_head:
            sentiment_out = module(sentiment_out)
        sentiment_probs = F.softmax(sentiment_out, dim=1)  # Take softmax along the logits dimension

        return classification_probs, sentiment_probs
