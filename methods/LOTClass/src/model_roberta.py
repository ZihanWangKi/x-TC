from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_roberta import RobertaLMHead
#from transformers.models.roberta.modeling_roberta import RobertaLMHead
from torch import nn
import sys


class LOTClassModel_roberta(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        #self.bert = self.roberta
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        # MLM head is not trained
        self.cls = self.lm_head
        for param in self.cls.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def forward(self, input_ids, pred_mode, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        bert_outputs = self.roberta(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds)
        last_hidden_states = bert_outputs[0]
        if pred_mode == "classification":
            trans_states = self.dense(last_hidden_states)
            trans_states = self.activation(trans_states)
            trans_states = self.dropout(trans_states)
            logits = self.classifier(trans_states)
        elif pred_mode == "mlm":
            logits = self.lm_head(last_hidden_states)
        else:
            sys.exit("Wrong pred_mode!")
        return logits
