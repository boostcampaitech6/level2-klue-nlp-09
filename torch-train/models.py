import transformers
import torch
import torch.nn as nn


def get_model(args):
    if args.model_type == 'auto':
        model = auto_model(args)
    elif args.model_type == 't5':
        model = T5Clf(args)
    elif args.model_type == 'lstm':
        model = AutoModelLSTMClassifier(args)
    elif args.model_type == 'entity_embedding':
        model = EntityEmbeddingCLF(args)
    elif args.model_type == 'pipeline_model':
        model = PipelineModel(args)
    elif args.model_type == 'aux':
        model = LMWithAuxiliaryCLF(args)
    return model



def auto_model(args, num_labels=30):
    print('Load Auto Sequence Classification Model')
    config = transformers.AutoConfig.from_pretrained(
        args.model_name, 
        cache_dir=args.cache_dir
    )
    config.num_labels = num_labels

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        config=config, 
        cache_dir=args.cache_dir
    )
    return model



class T5Clf(nn.Module):
    def __init__(self, args):
        super(T5Clf, self).__init__()
        self.args = args
        # self.t5 = transformers.T5EncoderModel.from_pretrained(
        self.t5 = transformers.T5Model.from_pretrained(
            args.model_name, cache_dir=args.cache_dir
        )

        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.t5.config.hidden_size, 30)
        self.config = self.t5.config
        self.config.num_labels = 30

        self._init_weights(self.classifier)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,labels=None):
        output = self.t5(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=decoder_input_ids,decoder_attention_mask=decoder_attention_mask)
        last_hidden_state = output.get('last_hidden_state')[:,0,:] # Decoder last token 위치의 attention [B, 512]
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        return {
            'logits':logits
        }

    # def forward(self, input_ids, attention_mask, labels=None):
    #     output = self.t5(input_ids=input_ids,attention_mask=attention_mask)
    #     last_hidden_state = output.get('last_hidden_state')[:,0,:] # Encoder first token 위치의 attention [B, 512]
    #     last_hidden_state = self.dropout(last_hidden_state)
    #     logits = self.classifier(last_hidden_state)
    #     return {
    #         'logits':logits
    #     }
    

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.5)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.5)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def resize_token_embeddings(self, nums):
        self.t5.resize_token_embeddings(nums)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))


class AutoModelLSTMClassifier(nn.Module):
    def __init__(self, args, num_layers=1, bidirectional=True):
        super(AutoModelLSTMClassifier, self).__init__()
        
        # 모델 로딩
        self.model = transformers.AutoModel.from_pretrained(
            args.model_name, cache_dir=args.cache_dir
        )
        self.config = self.model.config
        self.config.num_labels = 30
        
        # LSTM 레이어 정의
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        
        # 선형 분류 레이어 정의
        lstm_output_size = self.config.hidden_size * 2 if bidirectional else self.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(lstm_output_size, 30)

        self.init_weights()
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        model_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask) # [B, L, H]
        lstm_output, _ = self.lstm(model_output[0])
        lstm_output = self.dropout(lstm_output)
        logits = self.linear(lstm_output[:, -1, :])
        return {
            'logits':logits
        }
    
    def init_weights(self):
        self.linear.weight.data.normal_(mean=0.0, std=0.5)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def resize_token_embeddings(self, nums):
        self.model.resize_token_embeddings(nums)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))
        return self


class EntityEmbeddingCLF(nn.Module):
    def __init__(self, args):
        super(EntityEmbeddingCLF, self).__init__()
        
        self.args = args
        self.model = auto_model(args)

        self.entity_embeddings = nn.Embedding(1, self.model.config.hidden_size)
        self.entity_embeddings.weight.data.normal_(mean=0.0, std=0.5)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, entity_ids=None, labels=None):
        word_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)
        entity_embeds = self.entity_embeddings(entity_ids)
        inputs_embeds = word_embeds + entity_embeds

        outputs = self.model(
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        return outputs

    def resize_token_embeddings(self, nums):
        self.model.resize_token_embeddings(nums)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))
        return self



class PipelineModel(nn.Module):
    def __init__(self, args):
        super(PipelineModel, self).__init__()
        
        self.args = args
        self.model = auto_model(args) # dummy model for config
        self.config = self.model.config

        self.model1 = auto_model(args, num_labels=2) # no relation or relation binary-분류
        self.model2 = auto_model(args, num_labels=29) # 29 classes except no relation 분류 (29)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        batch = {
          'input_ids':input_ids,
          'token_type_ids':token_type_ids,
          'attention_mask':attention_mask
        }
        batch = {k:v for k,v in batch.items() if v is not None}
        logits1 = self.model1(**batch).get('logits') # [B, 2]
        logits2 = self.model2(**batch).get('logits') # [B, 29]
        logits2 = logits2 * logits1[:,1:]
        logits = torch.cat([logits1[:,:1], logits2], dim=1)
        return {
            'logits':logits
        }

    def resize_token_embeddings(self, nums):
        self.model1.resize_token_embeddings(nums)
        self.model2.resize_token_embeddings(nums)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))
        return self



class LMWithAuxiliaryCLF(nn.Module):
    def __init__(self, args):
        super(LMWithAuxiliaryCLF, self).__init__()
        
        self.args = args
        self.model = auto_model(args) # dummy model for config
        self.config = self.model.config

        self.auxiliary_embeddings = nn.Embedding(8, self.config.hidden_size)
        self.auxiliary_clf = nn.Sequential(
            nn.Conv1d(self.config.hidden_size, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 30, kernel_size=1),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4,1)
        ) # [B, 30, 1]

        self.main_weight = nn.Parameter(torch.randn(size=(1,)))
        self.sub_weight = nn.Parameter(torch.randn(size=(1,)))
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, type_ids=None, labels=None):
        batch = {
          'input_ids':input_ids,
          'token_type_ids':token_type_ids,
          'attention_mask':attention_mask
        }
        batch = {k:v for k,v in batch.items() if v is not None}
        logits = self.model(**batch).get('logits') # [B, 30]
        
        if self.training:
            # type_ids = [B, 4]
            aux_embeds = self.auxiliary_embeddings(type_ids) # [B, 4, H]
            aux_logits = self.auxiliary_clf(aux_embeds.transpose(1,2)).squeeze(-1) # [B, 30]

            logits = logits * self.main_weight + aux_logits * self.sub_weight

        return {
            'logits':logits
        }

    def resize_token_embeddings(self, nums):
        self.model.resize_token_embeddings(nums)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))
        return self






class StackingModel(nn.Module):
    def __init__(self, args):
        super(StackingModel, self).__init__()
        
        self.model_names = ['klue/roberta-large','xlm-roberta-large', 'team-lucid/deberta-v3-xlarge-korean']
        models = []
        for model_name in self.model_names:
            args.model_name = model_name
            models.append(get_model(args))
        
        self.config = models[0].config
        self.lms = nn.ModuleList(models)
        
        # Stacking Layer
        self.stackling_layer = nn.Linear()

        self.init_weights()
        
    def forward(self, texts, labels=None):
        bert_output = self.lm(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output[0])
        lstm_output = self.dropout(lstm_output)
        logits = self.linear(lstm_output[:, -1, :])
        # bert_output = self.dropout(bert_output[1])
        # logits = self.linear(bert_output)
        return {
            'logits':logits
        }

    def make_inputs(self, texts):
        for model_name in self.model_names:
            tokenizer(texts)
    
    def init_weights(self):
        self.linear.weight.data.normal_(mean=0.0, std=0.5)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def resize_token_embeddings(self, nums):
        self.lm.resize_token_embeddings(nums)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path))