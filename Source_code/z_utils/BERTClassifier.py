from transformers import AutoModel, AutoTokenizer
import torch


class BERTClassifier(torch.nn.Module):

    def __init__(self, checkpoint, dropout=0.1):

        super(BERTClassifier, self).__init__()

        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.bert = AutoModel.from_pretrained(checkpoint, num_labels=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        output = self.linear(dropout_output)

        return output
