from transformers import AutoModel, AutoTokenizer
import torch


class BlueBERTClassifier(torch.nn.Module):

    def __init__(self, checkpoint, dropout=0.1):

        super(BlueBERTClassifier, self).__init__()

        self.checkpoint = checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

        self.bluebert = AutoModel.from_pretrained(checkpoint, num_labels=2)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(self.bluebert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):

        bluebert_output = self.bluebert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        dropout_output = self.dropout(bluebert_output.pooler_output)
        output = self.linear(dropout_output)

        return output
