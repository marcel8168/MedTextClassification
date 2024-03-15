from transformers import DebertaForSequenceClassification, AutoTokenizer
import torch


class DeBERTaClassifier(torch.nn.Module):

    def __init__(self, checkpoint, dropout=0.1):

        super(DeBERTaClassifier, self).__init__()
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.deberta = DebertaForSequenceClassification.from_pretrained(
            checkpoint, num_labels=2)

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.deberta(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return output.logits
