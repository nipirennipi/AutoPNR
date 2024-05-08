import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import (
    pack_padded_sequence, 
    pad_packed_sequence, 
)
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, 
    BertModel, 
    BertOnlyMLMHead,
)


class BertAplo4NR(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        self.search_stage = False
        self.vocab_size = config.vocab_size
        self.mask_token_id = config.mask_token_id

        self.config = config
        self.prompt_tokens = torch.arange(config.prefix_size).long()
        self.prompt_generate = PromptGenerator(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def freeze(self):
        tuned_module_name = [
            'answer_search'
        ]
        for name, param in self.named_parameters():
            if not any(b in name for b in tuned_module_name):
                param.requires_grad = False

    def set_answer_search(self, need_search_module):
        self.search_stage = need_search_module
        self.answer_search = AnswerSearcher(self.config, need_search_module)

    def set_answer_dict(self, answer_dict):
        self.answer_search.set_answer_dict(answer_dict)

    def get_answer_dict(self):
        return self.answer_search.get_answer_dict()

    def forward(
            self,
            batch_input,
            batch_atten,
            batch_ccate,
            batch_hcate, 
            batch_hlen,
            batch_label,
        ):
        config = self.config
        batch_size = batch_input.shape[0]

        # Concat prompt attention mask
        if not self.search_stage:
            batch_patten = torch.ones(batch_size, config.prefix_size).to(batch_atten.device)
            batch_atten = torch.cat((batch_patten, batch_atten), dim=1)
        
        # Get prompt
        past_key_values = None
        if not self.search_stage:
            past_key_values = self.get_prompt(batch_size, batch_hcate, batch_hlen)

        # Feed the prompt and input to bert 
        outputs = self.bert(
            input_ids=batch_input, 
            attention_mask=batch_atten, 
            past_key_values=past_key_values, 
        )
        
        # Get <mask> token representation
        sequence_output = outputs[0]
        mask_pos = batch_input.eq(self.mask_token_id)
        mask_pos = torch.nonzero(mask_pos)
        sequence_mask_output = sequence_output[mask_pos[:, 0], mask_pos[:, 1], :]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Get answer
        answer_pair = self.answer_search(batch_ccate)
        
        # Return logits for each answer
        batch_logits = []
        for answer in answer_pair:
            batch_logits.append((prediction_mask_scores * answer).sum(dim=-1))
        batch_logits = torch.stack(batch_logits, dim=1)

        # CTR task
        batch_loss = None
        if batch_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            batch_loss = loss_fct(batch_logits, batch_label)

        batch_score = batch_logits.softmax(dim=1)[:, 1]
        return (batch_loss, batch_score) if batch_loss is not None else batch_score

    def get_prompt(self, batch_size, batch_hcate, batch_hlen):
        config = self.config
        past_key_values = self.prompt_generate(batch_hcate, batch_hlen)
        past_key_values = past_key_values.view(
            batch_size,
            config.prefix_size,
            config.num_hidden_layers * 2,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


class PromptGenerator(nn.Module):
    """
    reference:
        https://github.com/huggingface/peft/blob/main/src/peft/tuners/prefix_tuning/model.py#L21
        https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L835
    """
    def __init__(self, config):
        super(PromptGenerator, self).__init__()
        self.prefix_size = config.prefix_size 
        self.hidden_size = config.hidden_size

        n_cates = config.n_cates
        prefix_size = config.prefix_size
        generator_hidden_size = config.generator_hidden_size
        cate_dim = config.cate_dim
        hidden_size = config.hidden_size
        num_hidden_layers = config.num_hidden_layers
        gru_hidden_size = config.cate_dim
        gru_num_layers = config.gru_num_layers
        if gru_num_layers == 1:
            hidden_dropout_prob = 0
        else:
            hidden_dropout_prob = config.hidden_dropout_prob

        self.cate_embeddings = nn.Embedding(
            num_embeddings=n_cates + 1, 
            embedding_dim=cate_dim, 
            padding_idx=0, 
        )
        self.gru_head = nn.GRU(
            input_size=cate_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=hidden_dropout_prob,
        )
        self.prefix_head = nn.Sequential(
            nn.Linear(gru_hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * prefix_size),
        )
        # Use a two-layer MLP to encode the prefix
        
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, generator_hidden_size),
            nn.Tanh(),
            nn.Linear(generator_hidden_size, num_hidden_layers * 2 * hidden_size),
        )

    def forward(self, hcate, hlen):
        seq = self.cate_embeddings(hcate)
        packed_seq = pack_padded_sequence(
            input=seq, 
            lengths=hlen.cpu(), 
            batch_first=True, 
            enforce_sorted=False,
        )
        packed_out, _ = self.gru_head(packed_seq)
        unpack_out, unpack_len = pad_packed_sequence(
            sequence=packed_out, 
            batch_first=True
        )
        gru_out = unpack_out[range(len(unpack_len)), unpack_len - 1, :]
        prefix_tokens = self.prefix_head(gru_out)
        prefix_tokens = prefix_tokens.view(
            -1, self.prefix_size, self.hidden_size
        )
        past_key_values = self.transform(prefix_tokens)
        return past_key_values


class AnswerSearcher(nn.Module):
    def __init__(self, config, need_search_module):
        super(AnswerSearcher, self).__init__()
        self.vocab_size = config.vocab_size
        self.n_cates = config.n_cates
        self.search_stage = need_search_module
        
        vocab_size = config.vocab_size
        n_cates = config.n_cates
        cate_dim = config.cate_dim
        hidden_size = config.hidden_size
        n_labels = config.n_labels
        # Module for search stage
        if need_search_module:
            self.cate_embeddings = nn.Embedding(
                    num_embeddings=n_cates + 1,
                    embedding_dim=cate_dim,
                    padding_idx=0,
            )
            self.cate_verbalizer = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cate_dim, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, vocab_size),
                ) for _ in range(n_labels)
            ])
            self.gumbel = GumbelSoftmax(config)
        else:
            # Answer dict at [0] for nothing
            self.answer_dict = nn.ParameterDict({
                'dict': nn.Parameter(
                    torch.zeros(
                        size=(n_cates + 1, n_labels),
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                )
            })

    def forward(self, ccate):
        answer_pair = []
        # Search stage
        if self.search_stage:
            for verbalizer in self.cate_verbalizer:
                emb_out = self.cate_embeddings(ccate)
                vb_out = verbalizer(emb_out)
                answer = self.gumbel(vb_out)
                answer_pair.append(answer)
        # Retrain stage or inference stage
        else:
            answer = self.answer_dict['dict'].data[ccate].long()
            answer = F.one_hot(answer, num_classes=self.vocab_size)
            answer_pair = answer.chunk(2, 1)
            answer_pair = [answer.squeeze() for answer in answer_pair]
        return answer_pair

    def set_answer_dict(self, value):
        self.answer_dict['dict'].data = value

    def get_answer_dict(self):
        if not self.search_stage:
            return self.answer_dict['dict'].data
        
        answer_dict = []
        for verbalizer in self.cate_verbalizer:
            emb_out = self.cate_embeddings.weight.data
            vb_out = verbalizer(emb_out)
            answer = torch.argmax(vb_out, dim=-1)
            answer_dict.append(answer)
        answer_dict = torch.stack(answer_dict, dim=1)
        return answer_dict
        

class GumbelSoftmax(nn.Module):
    def __init__(self, config) -> None:
        super(GumbelSoftmax, self).__init__()
        self.tau = config.tau

    def forward(self, input, hard = True):
        output = F.gumbel_softmax(
            logits=input,
            tau=self.tau,
            hard=hard,
        )
        return output
