# ------------------------------------------------------------------------
# PDVC
# ------------------------------------------------------------------------
# Modified from Deformable DETR(https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
import math

from misc.detr_utils import box_ops
from misc.detr_utils.misc import (inverse_sigmoid)

from .matcher import build_matcher

from .deformable_transformer import build_deforamble_transformer
from pdvc.CaptioningHead import build_captioner
import copy
from .criterion import SetCriterion
# from .rl_tool import init_scorer
from misc.utils import decide_two_stage
from .base_encoder import build_base_encoder


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

      
class PDVC(nn.Module):
    """ This is the PDVC module that performs dense video captioning """

    def __init__(self, base_encoder, transformer, captioner, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, opt=None, translator=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            captioner: captioning head for generate a sentence for each event queries
            num_classes: number of foreground classes
            num_queries: number of event queries. This is the maximal number of events
                         PDVC can detect in a single video. For ActivityNet Captions, we recommend 10-30 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            opt: all configs
        """
        super().__init__()
        self.opt = opt
        self.base_encoder = base_encoder
        self.transformer = transformer
        # self.transformer.set_level_embed(self.base_encoder.hidden_size_list)
        self.caption_head = captioner

        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.count_head = nn.Linear(hidden_dim, opt.max_eseq_length + 1)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 2, 3)

        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.share_caption_head = opt.share_caption_head

        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        num_pred = transformer.decoder.num_layers
        if self.share_caption_head:
            print('all decoder layers share the same caption head')
            self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
        else:
            print('do NOT share the caption head')
            self.caption_head = _get_clones(self.caption_head, num_pred)

        if with_box_refine:
            self.class_head = _get_clones(self.class_head, num_pred)
            self.count_head = _get_clones(self.count_head, num_pred)
            self.bbox_head = _get_clones(self.bbox_head, num_pred)
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], -2)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_head = self.bbox_head
        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.class_head = nn.ModuleList([self.class_head for _ in range(num_pred)])
            self.count_head = nn.ModuleList([self.count_head for _ in range(num_pred)])
            self.bbox_head = nn.ModuleList([self.bbox_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = None

        self.translator = translator

        self.disable_mid_caption_heads = opt.disable_mid_caption_heads
        if self.disable_mid_caption_heads:
            print('only calculate caption loss in the last decoding layer')
        
        self.model_status = "vanilla PDVC\n"

    def get_filter_rule_for_encoder(self):
        filter_rule = lambda x: 'input_proj' in x \
                                or 'transformer.encoder' in x \
                                or 'transformer.level_embed' in x \
                                or 'base_encoder' in x
        return filter_rule

    def encoder_decoder_parameters(self):
        filter_rule = self.get_filter_rule_for_encoder()
        enc_paras = []
        dec_paras = []
        for name, para in self.named_parameters():
            if filter_rule(name):
                print('enc: {}'.format(name))
                enc_paras.append(para)
            else:
                print('dec: {}'.format(name))
                dec_paras.append(para)
        return enc_paras, dec_paras

    def forward(self, dt, criterion, transformer_input_type, eval_mode=False, data_type=None):

        vf = dt['video_tensor']  # (N, L, C)
        mask = ~ dt['video_mask']  # (N, L)
        duration = dt['video_length'][:, 1]
        N, L, C = vf.shape
        # assert N == 1, "batch size must be 1."

        srcs, masks, pos = self.base_encoder(vf, mask, duration)

        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.transformer.prepare_encoder_inputs(
            srcs, masks, pos)
        memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios,
                                                  lvl_pos_embed_flatten, mask_flatten)

        two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type,
                                                                                                dt, criterion)

        if two_stage:
            init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_proposal(
                proposals)
        else:
            query_embed = self.query_embed.weight
            proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
            init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory,
                                                                                                              query_embed)

        hs, inter_references = self.transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                level_start_index, valid_ratios, query_embed,
                                                                mask_flatten, proposals_mask, disable_iterative_refine)

        others = {'memory': memory,
                  'mask_flatten': mask_flatten,
                  'spatial_shapes': temporal_shapes,
                  'level_start_index': level_start_index,
                  'valid_ratios': valid_ratios,
                  'proposals_mask': proposals_mask}

        if eval_mode or self.opt.caption_loss_coef == 0:
            out, loss = self.parallel_prediction_full(dt, criterion, hs, init_reference, inter_references, others,
                                                      disable_iterative_refine)
        else:
            out, loss = self.parallel_prediction_matched(dt, criterion, hs, init_reference, inter_references, others,
                                                         disable_iterative_refine)
        return out, loss

    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim]
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0

    def parallel_prediction_full(self, dt, criterion, hs, init_reference, inter_references, others,
                                 disable_iterative_refine):
        outputs_classes = []
        outputs_classes0 = []
        outputs_coords = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []

        num_pred = hs.shape[0]
        for l_id in range(hs.shape[0]):
            if l_id == 0:
                reference = init_reference
            else:
                reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            output_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]

            # if self.opt.disable_mid_caption_heads and (l_id != hs.shape[0] - 1):
            if l_id != hs.shape[0] - 1:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid, reference, others, 'none')
            else:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid, reference, others, self.opt.caption_decoder_type)

            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            outputs_classes.append(outputs_class)
            outputs_classes0.append(output_count)
            outputs_coords.append(outputs_coord)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)
        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        output_count = torch.stack(outputs_classes0)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]

        all_out = {'pred_logits': outputs_class,
                   'pred_count': output_count,
                   'pred_boxes': outputs_coord,
                   'caption_probs': outputs_cap_probs,
                   'seq': outputs_cap_seqs}
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]

        loss, last_indices, aux_indices = criterion(out, dt['video_target'])
        return out, loss

    def parallel_prediction_matched(self, dt, criterion, hs, init_reference, inter_references, others,
                                    disable_iterative_refine):
        outputs_classes = []
        outputs_counts = []
        outputs_coords = []
        outputs_cap_costs = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []

        num_pred = hs.shape[0]
        for l_id in range(num_pred):
            hs_lid = hs[l_id]
            reference = init_reference if l_id == 0 else inter_references[
                l_id - 1]  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            outputs_count = self.predict_event_num(self.count_head[l_id], hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]

            cost_caption, loss_caption, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid,
                                                                                 reference, others, 'none')
            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            outputs_classes.append(outputs_class)
            outputs_counts.append(outputs_count)
            outputs_coords.append(outputs_coord)
            # outputs_cap_losses.append(cap_loss)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_count = torch.stack(outputs_counts)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]
        # outputs_cap_loss = torch.stack(outputs_cap_losses)

        all_out = {
            'pred_logits': outputs_class,
            'pred_count': outputs_count,
            'pred_boxes': outputs_coord,
            # 'caption_losses': outputs_cap_loss,
            'caption_probs': outputs_cap_probs,
            'seq': outputs_cap_seqs}
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            loss, last_indices, aux_indices = criterion(out, dt['video_target'])

            for l_id in range(hs.shape[0]):
                hs_lid = hs[l_id]
                reference = init_reference if l_id == 0 else inter_references[l_id - 1]
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference,
                                                                   others, self.opt.caption_decoder_type, indices)

                l_dict = {'loss_caption': cap_loss}
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)

            out.update({'caption_probs': cap_probs, 'seq': seq})
        else:
            loss, last_indices = criterion(out, dt['video_target'])

            l_id = hs.shape[0] - 1
            reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            indices = last_indices[0]
            cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid, reference,
                                                               others, self.opt.caption_decoder_type, indices)
            l_dict = {'loss_caption': cap_loss}
            loss.update(l_dict)

            out.pop('caption_losses')
            out.pop('caption_costs')
            out.update({'caption_probs': cap_probs, 'seq': seq})

        return out, loss

    def caption_prediction(self, cap_head, dt, hs, reference, others, captioner_type, indices=None):
        N_, N_q, C = hs.shape
        all_cap_num = len(dt['cap_tensor'])
        query_mask = others['proposals_mask']
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()

        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        if indices == None:
            row_idx, col_idx = 0, 0
            for i in range(N_):
                mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
                row_idx=row_idx + query_nums[i]
                col_idx= col_idx + gt_nums[i]

            bigids = mix_mask.nonzero(as_tuple=False)
            feat_bigids, cap_bigids = bigids[:, 0], bigids[:, 1]

        else:
            feat_bigids = torch.zeros(sum([len(_[0]) for _ in indices])).long()
            cap_bigids = torch.zeros_like(feat_bigids)
            total_query_ids = 0
            total_cap_ids = 0
            total_ids = 0
            max_pair_num = max([len(_[0]) for _ in indices])

            new_hr_for_dsa = torch.zeros(N_, max_pair_num, C)  # only for lstm-dsa
            cap_seq = dt['cap_tensor']
            new_seq_for_dsa = torch.zeros(N_, max_pair_num, cap_seq.shape[-1], dtype=cap_seq.dtype)  # only for lstm-dsa
            for i, index in enumerate(indices):
                feat_ids, cap_ids = index
                feat_bigids[total_ids: total_ids + len(feat_ids)] = total_query_ids + feat_ids
                cap_bigids[total_ids: total_ids + len(feat_ids)] = total_cap_ids + cap_ids
                new_hr_for_dsa[i, :len(feat_ids)] = hs[i, feat_ids]
                new_seq_for_dsa[i, :len(feat_ids)] = cap_seq[total_cap_ids + cap_ids]
                total_query_ids += query_nums[i]
                total_cap_ids += gt_nums[i]
                total_ids += len(feat_ids)
        cap_probs = {}
        flag = True

        if captioner_type == 'none':
            cost_caption = torch.zeros(N_, N_q, all_cap_num,
                                       device=hs.device)  # batch_size * num_queries * all_caption_num
            loss_caption = torch.zeros(N_, N_q, all_cap_num, device=hs.device)
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cost_caption, loss_caption, cap_probs, seq

        elif captioner_type in ['light']:
            clip = hs_r.unsqueeze(1)
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None

        elif self.opt.caption_decoder_type == 'standard':
            # assert N_ == 1, 'only support batchsize = 1'
            if self.training:
                seq = dt['cap_tensor'][cap_bigids]
                if self.opt.caption_cost_type != 'rl':
                    cap_prob = cap_head(hs[:, feat_bigids], reference[:, feat_bigids], others, seq)
                    cap_probs['cap_prob_train'] = cap_prob
            else:
                with torch.no_grad():
                    cap_prob = cap_head(hs[:, feat_bigids], reference[:, feat_bigids], others,
                                        dt['cap_tensor'][cap_bigids])
                    seq, cap_prob_eval = cap_head.sample(hs, reference, others)
                    if len(seq):
                        seq = seq.reshape(-1, N_q, seq.shape[-1])
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval

            flag = False
            pass

        if flag:
            clip_ext = clip[feat_bigids]
            clip_mask_ext = clip_mask[feat_bigids]

            if self.training:
                seq = dt['cap_tensor'][cap_bigids]
                if self.opt.caption_cost_type != 'rl':
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq)
                    cap_probs['cap_prob_train'] = cap_prob
            else:
                with torch.no_grad():
                    seq_gt = dt['cap_tensor'][cap_bigids]
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq_gt)
                    seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)

                    if len(seq):
                        # re_seq = torch.zeros(N_, N_q, seq.shape[-1])
                        # re_cap_prob_eval = torch.zeros(N_, N_q, cap_prob_eval.shape[-1])
                        seq = seq.reshape(-1, N_q, seq.shape[-1])
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval

        if self.opt.caption_cost_type == 'loss':
            cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1])
            caption_tensor = dt['cap_tensor'][:, 1:][cap_bigids]
            caption_mask = dt['cap_mask'][:, 1:][cap_bigids]
            cap_loss = cap_head.build_loss(cap_prob, caption_tensor, caption_mask)
            cap_cost = cap_loss

        else:
            raise AssertionError('caption cost type error')

        if indices:
            return cap_loss.mean(), cap_probs, seq

        cap_id, query_id = cap_bigids, feat_bigids
        cost_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        cost_caption[query_id, cap_id] = cap_cost
        loss_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        loss_caption[query_id, cap_id] = cap_loss
        cost_caption = cost_caption.reshape(-1, N_q,
                                            max(cap_id) + 1)  # batch_size * num_queries * all_caption_num
        loss_caption = loss_caption.reshape(-1, N_q, max(cap_id) + 1)
        return cost_caption, loss_caption, cap_probs, seq

    def caption_prediction_eval(self, cap_head, dt, hs, reference, others, decoder_type, indices=None):
        assert indices == None

        N_, N_q, C = hs.shape
        query_mask = others['proposals_mask']
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()
        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        row_idx, col_idx = 0, 0
        for i in range(N_):
            mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
            row_idx = row_idx + query_nums[i]
            col_idx = col_idx + gt_nums[i]

        cap_probs = {}

        if decoder_type in ['none']:
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cap_probs, seq

        elif decoder_type in ['light']:
            clip = hs_r.unsqueeze(1)
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None
            seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)
            if len(seq):
                seq = seq.reshape(-1, N_q, seq.shape[-1])
                cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
            cap_probs['cap_prob_eval'] = cap_prob_eval

        elif decoder_type in ['standard']:
            assert N_ == 1, 'only support batchsize > 1'
            with torch.no_grad():
                seq, cap_prob_eval = cap_head.sample(hs, reference, others)
                if len(seq):
                    seq = seq.reshape(-1, N_q, seq.shape[-1])
                    cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                cap_probs['cap_prob_eval'] = cap_prob_eval

        return cap_probs, seq


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @torch.no_grad()
    def forward(self, outputs, target_sizes, loader):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        N, N_q, N_class = out_logits.shape
        assert len(out_logits) == len(target_sizes)

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cl_to_xy(out_bbox)
        raw_boxes = copy.deepcopy(boxes)
        boxes[boxes < 0] = 0
        boxes[boxes > 1] = 1
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))

        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        seq = outputs['seq']  # [batch_size, num_queries, max_Cap_len=30]
        cap_prob = outputs['caption_probs']['cap_prob_eval']  # [batch_size, num_queries]
        eseq_lens = outputs['pred_count'].argmax(dim=-1).clamp(min=1)

        if len(seq):
            mask = (seq > 0).float()
            # cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float') / (
            #         1e-5 + mask.sum(2).cpu().numpy().astype('float'))
            cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
            seq = seq.detach().cpu().numpy().astype('int')  # (eseq_batch_size, eseq_len, cap_len)
            caps = [[loader.dataset.translator.rtranslate(s) for s in s_vid] for s_vid in seq]
            caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
            cap_scores = [[cap_scores[batch, idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
        else:
            bs, num_queries = boxes.shape[:2]
            cap_scores = [[-1e5] * num_queries] * bs
            caps = [[''] * num_queries] * bs

        results = [
            {'scores': s, 'labels': l, 'boxes': b, 'raw_boxes': b, 'captions': c, 'caption_scores': cs, 'query_id': qid,
             'vid_duration': ts, 'pred_seq_len': sl} for s, l, b, rb, c, cs, qid, ts, sl in
            zip(scores, labels, boxes, raw_boxes, caps, cap_scores, topk_boxes, target_sizes, eseq_lens)]
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_class=3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, n_class)
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.output(x)
        return x

class GradientReverse(Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    
def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)

class FeatureConvert(nn.Module):
    def __init__(self, input_dim, hidden_dim, opt):
        super().__init__()
        self.input_dim = input_dim
        self.pre_enc = self.pre_enc_large = None
        if len(input_dim) == 2:
            input_dim1, input_dim2 = input_dim
            self.input_dim = min(input_dim1, input_dim2)
            self.input_dim_large = max(input_dim1, input_dim2)            
            # adjust input feature size as the src and trg feature sizes may be different
            self.pre_enc_large = nn.Conv1d(self.input_dim_large, self.input_dim, kernel_size=1) 
            self.pre_enc = nn.Conv1d(self.input_dim, self.input_dim, kernel_size=1)
        
        # 1, 8192, 200
        self.enc_cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_dim * 4, kernel_size=1),
            nn.GroupNorm(32, hidden_dim * 4),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 2, kernel_size=1),
            nn.GroupNorm(32, hidden_dim * 2),
        )
        # 1, 512 * 2, 200        
        assert hasattr(opt, "n_view_cls"), "n_view_cls isn't specified" # self.cls_mlp = BinaryClassifier(hidden_dim * 2, hidden_dim)
        if opt.n_view_cls == 2:
            self.cls_mlp = BinaryClassifier(hidden_dim * 2, hidden_dim)
            self.n_view_cls = 2
        else:
            self.cls_mlp = MultiClassClassifier(hidden_dim * 2, hidden_dim, opt.n_view_cls)
            self.n_view_cls = 3            
            
        self.opt = opt
        self.prev_min_count = None

    def init_view_classifier(self):
        # init linear layer
        for m in self.cls_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y=None, is_train=True):                
        x = x.squeeze(0).transpose(1, 0).unsqueeze(0)
        assert x.shape[0] == 1, f"shape: {x.shape}"
        if self.pre_enc is not None and self.pre_enc_large is not None:
            if x.shape[1] == self.input_dim_large:
                x = self.pre_enc_large(x)
            elif x.shape[1] == self.input_dim:
                x = self.pre_enc(x)
            
        assert x.shape[1] == self.input_dim, f"shape: {x.shape}"
        out_feat = self.enc_cnn(x)
        frame_wise_feat = out_feat.squeeze(0).transpose(1, 0)  # 200, 512 * 2
        out_feat = out_feat.squeeze(0).transpose(1, 0).unsqueeze(0)
        if y is not None and (self.n_view_cls == 2 or not is_train): # only for pre-training with view classification            
            out_cls, cls_labels_sampled = self._view_classify(frame_wise_feat, y, is_train)
            return out_feat, out_cls, cls_labels_sampled
        return out_feat, None, None

    def _view_classify(self, frame_wise_feat, y, is_train=True):
        grl_coef = self.opt.view_grl_coef if is_train else 1.0
        is_grad_reverse = self.opt.grad_reverse == 'true' if is_train else False
        if is_train:
            frame_wise_feat_sampled, cls_labels_sampled = self._under_sample(frame_wise_feat, y)
        else:
            frame_wise_feat_sampled, cls_labels_sampled = frame_wise_feat, y
        if cls_labels_sampled is not None:
            if is_grad_reverse:
                out_cls = self.cls_mlp(grad_reverse(frame_wise_feat_sampled, scale=grl_coef))
            else:
                out_cls = self.cls_mlp(frame_wise_feat_sampled)
            # print("shape", out_feat.shape)
            return out_cls, cls_labels_sampled
        else:
            return None, None
            
    def _under_sample(self, x, y):
        unique_labels, counts = y.unique(return_counts=True)
        min_count = counts.min().item() # computed from the sample count for each class
        if len(unique_labels) == 1 and unique_labels[0] != 2:
            self.prev_min_count = None
            return None, None
        if len(unique_labels) == 2 and unique_labels[-1] == 2:
            self.prev_min_count = None
            return None, None
        under_sampled_x = torch.empty((0, x.shape[-1]), dtype=x.dtype).to(x.device)
        under_sampled_y = torch.empty((0,), dtype=y.dtype).to(y.device)
        
        # Under-sample each class
        for unique_label in unique_labels:
            if unique_label == 2: #samples is balanced from previous iteration
                # assert self.prev_min_count is not None, "min_count is filled from the previous inference"
                if self.prev_min_count is None:
                    return None, None
                min_count = self.prev_min_count
            else: # save min_count for the next iteration
                self.prev_min_count = min_count
            class_indices = (y == unique_label).nonzero(as_tuple=True)[0]
            class_count = class_indices.size(0)
            sampled_indices = class_indices[torch.randperm(class_count)[:min_count]]
            under_sampled_x = torch.cat([under_sampled_x, x[sampled_indices]], dim=0)
            under_sampled_y = torch.cat([under_sampled_y, y[sampled_indices]], dim=0)
                
        return under_sampled_x, under_sampled_y

class PDVCViewClass(PDVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_status = "PDVC view class model\n"
        input_dim = [self.opt.feature_dim_src, self.opt.feature_dim_trg]
        self.feat_convert = FeatureConvert(input_dim, self.opt.hidden_dim, self.opt)
        self.model_status += f"Using feature converter: (input: {input_dim})\n"
        
        if self.feat_convert is not None:
            self.n_view_cls = self.feat_convert.n_view_cls
            
        self.model_status += f"n_view_cls: {self.n_view_cls}\n"
        if self.n_view_cls == 2:
            self.view_loss = nn.BCELoss(reduction='none')  # binary cross entropy
        elif self.n_view_cls == 3:
            self.view_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.view_loss = None
    
    def view_classify(self, out_feat, y, is_train=True):
        frame_wise_feat = out_feat.squeeze(0) #.transpose(1, 0)
        out_view_cls, gt_view_labels_sampled = self.feat_convert._view_classify(frame_wise_feat, y, is_train)
        if out_view_cls is None:
            loss_view = torch.zeros(1).to(y.device)
            return loss_view, -1, -1
        else:
            loss_view = self.view_loss(out_view_cls, gt_view_labels_sampled.squeeze(-1)) #* gt_view_labels > 0
            loss_view = loss_view.mean()
            # for debug
            self.n_frame_feat_sampled = gt_view_labels_sampled.size(0)
            # predictions
            preds = torch.argmax(out_view_cls.detach().cpu(), dim=1)
            # calculate accuracy
            correct = (preds == gt_view_labels_sampled.cpu().view(-1)).float() # convert into float for division 
            view_acc = correct.sum()
            n_frame_feat = len(correct)
            view_acc = view_acc.item()
            
        return loss_view, view_acc, n_frame_feat
    
    def forward(self, dt, criterion, transformer_input_type, eval_mode=False, data_type=None):

        vf = dt['video_tensor']  # (N, L, C)
        mask = ~ dt['video_mask']  # (N, L)
        duration = dt['video_length'][:, 1]
        N, L, C = vf.shape
        # assert N == 1, "batch size must be 1."
        if dt['gt_view_labels'] is not None:
            gt_view_labels = dt['gt_view_labels'].transpose(1, 0)
            assert gt_view_labels.shape[1] == 1, f"{gt_view_labels.shape}"   
        else:
            gt_view_labels = None

        out_feat, out_view_cls, gt_view_labels_sampled = self.feat_convert(vf, gt_view_labels, is_train=not eval_mode)                        
        srcs, masks, pos = self.base_encoder(out_feat, mask, duration)
        view_acc = view_f1 = -1
        loss_view = torch.zeros(1).to(vf.device)
        if self.opt.view_loss_coef > 0:
            if self.n_view_cls == 2: # only for pre-training with view classification
                if out_view_cls is not None:                    
                    gt_view_labels_sampled = gt_view_labels_sampled.float()
                    loss_view = self.view_loss(out_view_cls, gt_view_labels_sampled.view(-1, 1)) #.squeeze(-1)) #* gt_view_labels > 0
                    loss_view = loss_view.mean()
                    # predictions
                    preds = torch.argmax(out_view_cls.detach().cpu(), dim=1)
                    # calculate accuracy
                    correct = (preds == gt_view_labels_sampled.cpu().view(-1)).float() # convert into float for division 
                    view_acc = correct.sum()
                    n_frame_feat = len(correct)
                    view_acc = view_acc.item()
            
        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.transformer.prepare_encoder_inputs(
            srcs, masks, pos)
        memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios,
                                                  lvl_pos_embed_flatten, mask_flatten)

        two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type,
                                                                                                dt, criterion)

        if two_stage:
            init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_proposal(
                proposals)
        else:
            query_embed = self.query_embed.weight
            proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
            init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory,
                                                                                                              query_embed)

        hs, inter_references = self.transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                level_start_index, valid_ratios, query_embed,
                                                                mask_flatten, proposals_mask, disable_iterative_refine)

        others = {'memory': memory,
                  'mask_flatten': mask_flatten,
                  'spatial_shapes': temporal_shapes,
                  'level_start_index': level_start_index,
                  'valid_ratios': valid_ratios,
                  'proposals_mask': proposals_mask,                  
                  }

        if eval_mode or self.opt.caption_loss_coef == 0:
            out, loss = self.parallel_prediction_full(dt, criterion, hs, init_reference, inter_references, others,
                                                      disable_iterative_refine)
        else:
            out, loss = self.parallel_prediction_matched(dt, criterion, hs, init_reference, inter_references, others,
                                                         disable_iterative_refine)
        if self.opt.view_loss_coef > 0:
            if self.n_view_cls == 2 or eval_mode:
                loss.update({                   
                                "loss_view": loss_view,
                            })
                out.update({
                    "out_feat": out_feat,
                    "view_labels": gt_view_labels,
                })
            else:
                out.update({
                    "out_feat": out_feat,
                    "view_labels": gt_view_labels,
                })
            out['view_acc'] = float(view_acc)
        return out, loss


def build(args):
    device = torch.device(args.device)    
    transformer = build_deforamble_transformer(args)
    captioner = build_captioner(args)

    if args.model_type == "pdvc":
        base_encoder = build_base_encoder(args)
        model = PDVC(
            base_encoder,
            transformer,
            captioner,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            opt=args
        )
    elif args.model_type == "pdvc-view":
        print("loading pdvc view model")
        base_encoder = build_base_encoder(args, feature_dim=args.hidden_dim * 2)
        model = PDVCViewClass(
            base_encoder=base_encoder,
            transformer=transformer,
            captioner=captioner,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            opt=args
        )
    else:
        raise NotImplementedError()

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,
                   'loss_counter': args.count_loss_coef,
                   'loss_caption': args.caption_loss_coef,
                   'loss_view': args.view_loss_coef,
                   }

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             focal_gamma=args.focal_gamma, opt=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args)}

    return model, criterion, postprocessors
