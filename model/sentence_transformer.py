import torch.nn as nn
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
