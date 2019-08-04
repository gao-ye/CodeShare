import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from PIL import Image
import time
import cv2
import os,pdb

from .transformer.Beam import Beam
from .transformer.Decoder import Decoder
from .transformer.Constants import UNK,PAD,BOS,EOS,PAD_WORD,UNK_WORD,BOS_WORD,EOS_WORD
from .resnet import resnet34, extract_g
im_count = 0
if (os.path.exists('attn_imgs') == False): 
    os.makedirs('attn_imgs')

class MODEL(nn.Module):

    def __init__(self, n_bm, n_vocab, 
        inputDataType='torch.cuda.FloatTensor', maxBatch=256, tgt_emb_prj_weight_sharing=False):
        super(MODEL, self).__init__()
        self.device = torch.device('cuda')
        self.n_bm = n_bm
        d_model = 1024
        d_word_vec = 512
        self.max_seq_len = 100
        n_tgt_vocab = n_vocab + 4  # add BOS EOS PAD UNK
        self.encoder = resnet34(pretrained=True)
        self.extract_g = extract_g(512)
        self.conv1x1 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        
        #self.src_position_enc = nn.Embedding(15, d_word_vec, padding_idx=0)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=self.max_seq_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=2048,
            n_layers=1, n_head=16, d_k=64, d_v=64,
            dropout=0.1)
        
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.
    def beam_decode_step(self,inst_dec_beams, len_dec_seq, src_seq, enc_output, global_feat, inst_idx_to_position_map, n_bm):

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, src_seq, enc_output, global_feat, n_active_inst, n_bm):
            dec_output, slf_attns, enc_attns = self.decoder(dec_seq, dec_pos, src_seq, enc_output, global_feat, return_attns=True,if_test=True)
            dec_output = dec_output[:, -1, :] 
                          
            dec_output_prj = self.tgt_word_prj(dec_output)
            
            word_prob = F.log_softmax(dec_output_prj*self.x_logit_scale, dim=1)
            word_prob = word_prob.view(n_active_inst, n_bm, -1)

            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map): 
            
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]
            
            return active_inst_idx_list
        
        n_active_inst = len(inst_idx_to_position_map)      
        
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
        if len_dec_seq==1:
           for i in range(len(dec_seq)):
               dec_seq[i][0] = BOS
               
        sorted_map = sorted(inst_idx_to_position_map.items(),key = lambda x:x[1])
        active_index = [ori_indx for ori_indx,new_indx in sorted_map]
        global_feat = global_feat[active_index]
        
        word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, global_feat, n_active_inst, n_bm)
        
        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_prob, inst_idx_to_position_map)

        return active_inst_idx_list
    def collect_hypothesis_and_scores(self,inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]
            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores
    def get_inst_idx_to_tensor_position_map(self, inst_idx_list):
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}
    def collect_active_part(self, beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''
        n_curr_active_inst = len(curr_active_inst_idx)
        if len(beamed_tensor.size())==2:
           _, d_hs = beamed_tensor.size()
           new_shape = (n_curr_active_inst * n_bm, d_hs)
        elif len(beamed_tensor.size())==3:
           _, d_hs1, d_hs2 = beamed_tensor.size()
           new_shape = (n_curr_active_inst * n_bm, d_hs1, d_hs2)
        else:
           print(beamed_tensor.size())
           assert 0==1
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(new_shape)
        return beamed_tensor
    def collate_active_info(self,src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list, n_bm):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)

        
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx.sort()
        #print(active_inst_idx)
        active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)
        
        active_src_seq = self.collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
        active_src_enc = self.collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)

        active_inst_idx_list.sort()
        
        #active_inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        active_inst_idx_to_position_map = {}
        count = 0
        for idx in active_inst_idx_list:
            active_inst_idx_to_position_map[idx] = count
            count += 1
        return active_src_seq, active_src_enc, active_inst_idx_to_position_map
            
    def forward(self, x, length, text, test=False, get_attention=False,cpu_texts=None):
        N = x.shape[0]
        if len(x.shape)==4 and x.shape[1]==1:
           x = x.expand(x.shape[0],3,x.shape[2],x.shape[3])
        if torch.max(length) > self.max_seq_len-1:
           print(cpu_texts)
           print(length)
           assert torch.max(length) <= self.max_seq_len-1         

        # Encoder #
        cnn_feat = self.encoder(x)
        global_feat = self.extract_g(cnn_feat)
        cnn_feat = self.conv1x1(cnn_feat)
        cnn_feat = cnn_feat.squeeze(-1).permute(0, 2, 3, 1).contiguous().view(N, -1, cnn_feat.shape[1])
        # Decoder #
                
        tgt_seq = torch.ones(N,self.max_seq_len).long().cuda() * PAD
        tgt_pos = torch.zeros(N,self.max_seq_len).long().cuda()
        tgt_seq[:,0] = BOS
        text_pos = 0
        for i in range(N):
            tgt_seq[i,1:length[i]+1] = text[text_pos:text_pos+length[i]]
            text_pos += length[i]
        
        for i in range(N):
            for j in range(length[i]+1): # BOS
                tgt_pos[i][j] = j+1        
        
        if test==False:
           dec_output, slf_attns, enc_attns = self.decoder(tgt_seq, tgt_pos, None, cnn_feat, global_feat, return_attns=True)
                     
           global im_count
           if torch.cuda.current_device()==0:
              alphabet='0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z ! " \' # $ % & ( ) * + , - . / : ; < = > ? @ [ \\ ] _ ` ~'
              alphabet=alphabet.split(' ')
              im_mean = [0.5, 0.5, 0.5]
              im_std = [0.5, 0.5, 0.5]
              
              for rand_index in range(N):
                 
                 im = x[rand_index].cpu().numpy()  # C,H,W
                 im = im.transpose(1,2,0)
                 im = im * np.array(im_std).astype(float)
                 im = im + np.array(im_mean)
                 im = im * 255.
                 im = np.clip(im,0,255)
                 im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
                 h, w, _ = im.shape
                    
                 for b in range(len(enc_attns)):
                    attns = enc_attns[b]
                    attns = attns.reshape(-1,N,self.max_seq_len,cnn_feat.shape[1])
                    attn = torch.sum(attns[:,rand_index,:,:],0)
                    for i in tgt_pos[rand_index].data.cpu().numpy():
                       if i >0 and tgt_seq[rand_index][i]<BOS:
                          score = attn[i-1]
                          assert cnn_feat.shape[1]==4*13
                          coff = score.data.cpu().numpy().reshape((4,13))
                          coff =  (coff - coff.min()) / (coff.max() - coff.min() + 1e-5)
                          assert (coff.max() - coff.min() + 1e-5)> 0
                          coff = np.uint8(255 * coff)
                          heatmap = cv2.applyColorMap(cv2.resize(coff, (w, h)), cv2.COLORMAP_JET)                       
                          result = heatmap * 0.3 + im * 0.6
                          
                          cv2.imwrite('attn_imgs/img_'+str(im_count)+'_char_'+str(i-1)+alphabet[tgt_seq[rand_index][i]]+'.jpg', result)
           
                 im_count += 1
                     
           seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
           seq_logit = seq_logit.view(-1, seq_logit.size(2))
           for i,lenth in enumerate(length):
               if i==0:
                  seq_stacked = seq_logit[0:lenth]
               else:
                  seq_stacked = torch.cat((seq_stacked,seq_logit[i*self.max_seq_len:i*self.max_seq_len+lenth]),0)
        else:
            src_enc = cnn_feat
            n_inst, len_s, d_h = src_enc.shape
            src_seq = torch.ones((n_inst,len_s)).long().cuda()
            assert n_inst==N
            src_seq = src_seq.repeat(1, self.n_bm).view(n_inst * self.n_bm, len_s)
            src_enc = src_enc.repeat(1, self.n_bm, 1).view(n_inst * self.n_bm, len_s, d_h)
            inst_dec_beams = [Beam(self.n_bm, device=self.device) for _ in range(n_inst)]
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            
            #-- Decode
            for len_dec_seq in range(1, self.max_seq_len):
                active_inst_idx_list = self.beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, src_enc, global_feat, inst_idx_to_position_map, self.n_bm)
                if not active_inst_idx_list:
                    break
 
                src_seq, src_enc, inst_idx_to_position_map = self.collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list,self.n_bm)
            
            batch_hyp, batch_scores = self.collect_hypothesis_and_scores(inst_dec_beams, 1)
                    
            seq_stacked = []
            for i,lenth in enumerate(length):
                old_len = len(seq_stacked)
                lenth_add5eos = lenth + 5
                if len(batch_hyp[i][0])>=lenth_add5eos:
                   seq_stacked.extend(batch_hyp[i][0][0:lenth_add5eos])
                else:
                   pad_num = lenth_add5eos - len(batch_hyp[i][0])
                   seq_stacked.extend(batch_hyp[i][0])
                   for pad_i in range(pad_num):
                       seq_stacked.extend([EOS])
                #assert len(cpu_texts[i])+5==(len(seq_stacked)-old_len)
            seq_stacked = torch.Tensor(seq_stacked).long().cuda()


        if get_attention:
           return seq_stacked, slf_attns, enc_attns
        if test==True:
           return seq_stacked, batch_scores
        else:
           return seq_stacked
        