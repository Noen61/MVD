import torch
from torch import nn
from typing import Dict,Optional

class RetrievalModel(nn.Module):

    def __init__(
        self, mention_encoder,entity_encoder,device ="cuda"
    ):
        super(RetrievalModel, self).__init__()

        self.mention_encoder  = mention_encoder
        self.entity_encoder = entity_encoder
        self.device = device

    def forward(self, mention_ids:Optional[torch.LongTensor]=None,entity_ids:Optional[torch.LongTensor]=None):
        if mention_ids is None:
            entity_mask = 1 - (entity_ids == 0).long()
            entity_emd = self.entity_encoder(input_ids=entity_ids,attention_mask=entity_mask)[0][:,0,:]
            return entity_emd
        if entity_ids is None:
            mention_mask = 1 - (mention_ids == 0).long()
            mention_emd = self.mention_encoder(input_ids=mention_ids,attention_mask=mention_mask)[0][:,0,:]
            return mention_emd

        mention_mask = 1 - (mention_ids == 0).long()
        mention_emd = self.mention_encoder(input_ids=mention_ids,attention_mask=mention_mask)[0][:,0,:]
        if len(entity_ids.size()) == 3:
            B,S,_ = entity_ids.size()
            entity_ids = entity_ids.view(B*S,-1)
            entity_mask = 1 - (entity_ids == 0).long()
            entity_emd = self.entity_encoder(input_ids=entity_ids,attention_mask=entity_mask)[0][:,0,:]
            logits = torch.matmul(entity_emd,mention_emd.T)
            logits = logits.view(B,S,B).permute(2, 0, 1).max(dim=-1)[0]
            labels = torch.arange(mention_emd.size(0)).long().to(self.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        else:
            loss = None
            # (batch_size,candidates_num,views_num,input_length)
            B,C,S,L = entity_ids.size()
            entity_ids = entity_ids.view(B*C*S,-1)
            entity_mask = 1 - (entity_ids == 0).long()
            entity_emd = self.entity_encoder(input_ids=entity_ids,attention_mask=entity_mask)[0][:,0,:]
            entity_emd = entity_emd.view(B,C*S,-1)
            # dot-product
            logits = torch.matmul(entity_emd,mention_emd.unsqueeze(-1))
            logits = logits.squeeze(dim=-1).view(B,C,S)

        return loss,logits

class TeacherModel(nn.Module):

    def __init__(
        self, rank_encoder,device ="cuda"
    ):
        super(TeacherModel, self).__init__()

        self.rank_encoder = rank_encoder
        self.ranker_classifier = nn.Linear(self.rank_encoder.config.hidden_size, 1)
        self.device = device

    def forward(self, input_ids:torch.LongTensor,segment_ids:torch.LongTensor,label:Optional[torch.LongTensor]=None,return_loss=True):
        if len(input_ids.size()) == 3:
            B,S,L = input_ids.size()
            input_ids = input_ids.view(B*S,-1)
            input_mask = 1 - (input_ids == 0).long()
            segment_ids = segment_ids.view(B*S).unsqueeze(0).expand(-1,L)
            token_type_ids = torch.arange(L).unsqueeze(0).expand(B*S,-1).to(segment_ids.device)
            token_type_ids = 1 - (token_type_ids<segment_ids).long()
            model_output = self.rank_encoder(input_ids=input_ids,attention_mask=input_mask,token_type_ids=token_type_ids)[0][:,0,:]
            model_output = model_output.view(B,S,-1)
            logits = self.ranker_classifier(model_output)
            logits = logits.squeeze(dim=-1)
        else:
            B,C,S,L = input_ids.size()
            input_ids = input_ids.view(B*C*S,-1)
            input_mask = 1 - (input_ids == 0).long()
            segment_ids = segment_ids.view(B*C*S).unsqueeze(-1).expand(-1,L)
            token_type_ids = torch.arange(L).unsqueeze(0).expand(B*C*S,-1).to(segment_ids.device)
            token_type_ids = 1 - (token_type_ids<segment_ids).long()
            model_output = self.rank_encoder(input_ids=input_ids,attention_mask=input_mask,token_type_ids=token_type_ids)[0][:,0,:]
            model_output = model_output.view(B,C,S,-1)
            #(B,C,S)
            logits = self.ranker_classifier(model_output).squeeze(dim=-1)
            if return_loss == False:
                return logits

        loss = 0
        if label is not None:
            #(B,C)
            logits = logits.max(dim=-1)[0]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label)

        return loss,logits

class MVD(nn.Module):

    def __init__(
        self, retriever,teacher,device ="cuda"
    ):
        super(MVD, self).__init__()

        self.retriever  = retriever
        self.teacher = teacher

        self.alpha = 0.3
        self.beta = 0.1

        self.device = device

    def forward(self, mention_ids:torch.LongTensor=None,entity_ids:torch.LongTensor=None,input_ids:torch.LongTensor=None,segment_ids:torch.LongTensor=None,label:Optional[torch.LongTensor]=None):
        # Inference
        if input_ids is None:
            if mention_ids is None:
                entity_emd = self.retriever(entity_ids=entity_ids)
                return entity_emd
            elif entity_ids is None:
                mention_emd = self.retriever(mention_ids=mention_ids)
                return mention_emd
            
        B,C,S,L = input_ids.size()
        _,retrieval_logits = self.retriever(mention_ids,entity_ids)
        teacher_logits = self.teacher(input_ids,segment_ids,return_loss=False)
        # index of relevent-view selected by teacher
        relevent_view_index = teacher_logits.argmax(dim=-1).unsqueeze(-1)

        #(B,C) align the relevant-view
        retrieval_logits_aligned = retrieval_logits.gather(index=relevent_view_index,dim=-1).squeeze(dim=-1)
        # w/o align the relevant-view
        retrieval_logits_origin = retrieval_logits.max(dim=-1)[0]
        teacher_logits_origin = teacher_logits.max(dim=-1)[0]

        loss_fct = nn.CrossEntropyLoss()
        kl_loss_fct = nn.KLDivLoss(reduction='batchmean')

        # supervised NCE loss
        retrieval_loss = loss_fct(retrieval_logits_origin, label)
        teacher_loss = loss_fct(teacher_logits_origin, label)
        # distill score distribution among candidate entities
        cross_alignment_loss = kl_loss_fct(retrieval_logits_aligned.log_softmax(-1),teacher_logits_origin.softmax(-1))
        # distill score distribution within each candidate entity
        self_alignment_loss = kl_loss_fct(retrieval_logits.view(B*C,-1).log_softmax(-1),teacher_logits.view(B*C,-1).softmax(-1))
        loss = retrieval_loss + teacher_loss + self.alpha*cross_alignment_loss+self.beta*self_alignment_loss
        
        return loss



