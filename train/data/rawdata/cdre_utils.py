import pdb
import math
import json
import random
import logging
import torch
from itertools import combinations
from pycdre.buffer import Buffer
from pycdre.sentence_reordering import SentReOrdering
from pycdre.buffer import BLOCK_SIZE


CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102
H_START_MARKER_ID = 1
H_END_MARKER_ID = 2
T_START_MARKER_ID = 3
T_END_MARKER_ID = 4


def gen_c(passage, span, bound_tokens, d_start, d_end):
    MAXLEN = 99999

    ret = []
    ret.append(bound_tokens[0])
    for i in range(span[0], span[1]):
        ret.append(passage[i])
    ret.append(bound_tokens[1])

    prev = []
    prev_ptr = span[0] - 1
    while prev_ptr >= 0 and len(prev) < MAXLEN:
        if prev_ptr in d_end:
            prev.append(f'[unused{(d_end[prev_ptr] + 2) * 2 + 2}]')
        prev.append(passage[prev_ptr])
        if prev_ptr in d_start:
            prev.append(f'[unused{(d_start[prev_ptr] + 2) * 2 + 1}]')
        prev_ptr -= 1
    prev.reverse() 
    
    nex = []
    nex_ptr = span[1]
    while nex_ptr < len(passage) and len(nex) < MAXLEN:
        if nex_ptr in d_start:
            nex.append(f'[unused{(d_start[nex_ptr] + 2) * 2 + 1}]')
        nex.append(passage[nex_ptr])
        if nex_ptr in d_end:
            nex.append(f'[unused{(d_end[nex_ptr] + 2) * 2 + 2}]') 
        nex_ptr += 1
    
    return prev + ret + nex


def check_htb(input_ids, h_t_flag):
    htb_mask_list = []
    htb_list_batch = []
    for pi in range(input_ids.size()[0]):
        # pdb.set_trace()
        tmp = torch.nonzero(
            input_ids[pi] - torch.full(([input_ids.size()[1]]), 1).to(input_ids.device))
        if tmp.size()[0] < input_ids.size()[0]:
            print(input_ids)
        try:
            h_starts = [i[0]
                        for i in (input_ids[pi] == 1).nonzero().detach().tolist()]
            h_ends = [i[0]
                      for i in (input_ids[pi] == 2).nonzero().detach().tolist()]
            t_starts = [i[0]
                        for i in (input_ids[pi] == 3).nonzero().detach().tolist()]
            t_ends = [i[0]
                      for i in (input_ids[pi] == 4).nonzero().detach().tolist()]
            if len(h_starts) == len(h_ends):
                h_start = h_starts[0]
                h_end = h_ends[0]
            else:
                for h_s in h_starts:
                    for h_e in h_ends:
                        if 0 < h_e - h_s < 20:
                            h_start = h_s
                            h_end = h_e
            if len(t_starts) == len(t_ends):
                t_start = t_starts[0]
                t_end = t_ends[0]
            else:
                for t_s in t_starts:
                    for t_e in t_ends:
                        if 0 < t_e - t_s < 20:
                            t_start = t_s
                            t_end = t_e
            if h_end-h_start <= 0 or t_end-t_start <= 0:
                print("H/T INDEX ERROR")
                print(h_starts)
                print(h_ends)
                print(t_starts)
                print(t_ends)
                # pdb.set_trace()
                if h_end-h_start <= 0:
                    for h_s in h_starts:
                        for h_e in h_ends:
                            if 0 < h_e - h_s < 20:
                                h_start = h_s
                                h_end = h_e
                if t_end-t_start <= 0:
                    for t_s in t_starts:
                        for t_e in t_ends:
                            if 0 < t_e - t_s < 20:
                                t_start = t_s
                                t_end = t_e

                if h_end-h_start <= 0 or t_end-t_start <= 0:
                    pdb.set_trace()

            b_spans = torch.nonzero(torch.gt(torch.full(([input_ids.size()[1]]), 99).to(
                input_ids.device), input_ids[pi])).squeeze(0).squeeze(1).detach().tolist()
            token_len = input_ids[pi].nonzero().size()[0]
            b_spans = [i for i in b_spans if i <= token_len-1]
            assert len(b_spans) >= 4
            # for i in [h_start, h_end, t_start, t_end]:
            for i in h_starts + h_ends + t_starts + t_ends:
                b_spans.remove(i)
            h_span = [h_pos for h_pos in range(h_start, h_end+1)]
            t_span = [t_pos for t_pos in range(t_start, t_end+1)]
            h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(
                0, torch.tensor(h_span).to(input_ids.device), 1)
            t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(
                0, torch.tensor(t_span).to(input_ids.device), 1)
        except:
            # pdb.set_trace()
            h_span = []
            t_span = []
            h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device)
            t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device)
            b_spans = []
        b_span_ = []
        if len(b_spans) > 0 and len(b_spans) % 2 == 0:
            b_span_chunks = [b_spans[i:i+2] for i in range(0, len(b_spans), 2)]
            b_span = []
            for span in b_span_chunks:
                b_span.extend([b_pos for b_pos in range(span[0], span[1]+1)])
            b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(
                0, torch.tensor(b_span).to(input_ids.device), 1)
            b_span_.extend(b_span)
        elif len(b_spans) > 0 and len(b_spans) % 2 == 1:
            b_span = []
            ptr = 0
            # pdb.set_trace()
            while (ptr <= len(b_spans)-1):
                try:
                    if input_ids[pi][b_spans[ptr+1]] - input_ids[pi][b_spans[ptr]] == 1:
                        b_span.append([b_spans[ptr], b_spans[ptr+1]])
                        ptr += 2
                    else:
                        ptr += 1
                except IndexError as e:
                    # pdb.set_trace()
                    ptr += 1
            for bs in b_span:
                # pdb.set_trace()
                # ex_bs = range(bs[0], bs[1])
                b_span_.extend(bs)
                if len(b_span_) % 2 != 0:
                    print(b_spans)
            b_span_chunks = [b_span_[i:i+2] for i in range(0, len(b_span_), 2)]
            b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(
                0, torch.tensor(b_span_).to(input_ids.device), 1)
        else:
            b_span_ = []
            b_span_chunks = []
            b_mask = torch.zeros_like(input_ids[pi])
        htb_mask = torch.concat([h_mask.unsqueeze(0), t_mask.unsqueeze(
            0), b_mask.unsqueeze(0)], dim=0)  # [3,512]
        htb_mask_list.append(htb_mask)
        htb_list_batch.append([h_span, t_span, b_span_chunks])
    bag_len = input_ids.size()[0]
    for dp in range(0, bag_len):
        try:
            h_span = htb_list_batch[dp][0]  # [3,4,5,6,7]
            t_span = htb_list_batch[dp][1]  # [71,72,73,74]
            if h_span == [] or t_span == []:
                # print("fail detecting h/t")
                # h_t_flag = False
                # pdb.set_trace()
                # flag do not change
                pass
            else:
                # pdb.set_trace()
                h_t_flag = True
                # print("H/T Detected")
        except Exception as e:
            print(e)
            pdb.set_trace()
    # entity_mask = T[8,3,512] h_mask = T[8, 512] t_mask = T[8,512] b_mask = T[8,512]
    return h_t_flag


def check_htb_debug(input_ids, h_t_flag):
    htb_mask_list = []
    htb_list_batch = []
    pdb.set_trace()
    for pi in range(input_ids.size()[0]):
        # pdb.set_trace()
        tmp = torch.nonzero(
            input_ids[pi] - torch.full(([input_ids.size()[1]]), 1).to(input_ids.device))
        if tmp.size()[0] < input_ids.size()[0]:
            print(input_ids)
        try:
            h_starts = [i[0]
                        for i in (input_ids[pi] == 1).nonzero().detach().tolist()]
            h_ends = [i[0]
                      for i in (input_ids[pi] == 2).nonzero().detach().tolist()]
            t_starts = [i[0]
                        for i in (input_ids[pi] == 3).nonzero().detach().tolist()]
            t_ends = [i[0]
                      for i in (input_ids[pi] == 4).nonzero().detach().tolist()]
            if len(h_starts) == len(h_ends):
                h_start = h_starts[0]
                h_end = h_ends[0]
            else:
                for h_s in h_starts:
                    for h_e in h_ends:
                        if 0 < h_e - h_s < 20:
                            h_start = h_s
                            h_end = h_e
            if len(t_starts) == len(t_ends):
                t_start = t_starts[0]
                t_end = t_ends[0]
            else:
                for t_s in t_starts:
                    for t_e in t_ends:
                        if 0 < t_e - t_s < 20:
                            t_start = t_s
                            t_end = t_e
            if h_end-h_start <= 0 or t_end-t_start <= 0:
                print("H/T INDEX ERROR")
                print(h_starts)
                print(h_ends)
                print(t_starts)
                print(t_ends)
                pdb.set_trace()
                if h_end-h_start <= 0:
                    for h_s in h_starts:
                        for h_e in h_ends:
                            if 0 < h_e - h_s < 20:
                                h_start = h_s
                                h_end = h_e
                if t_end-t_start <= 0:
                    for t_s in t_starts:
                        for t_e in t_ends:
                            if 0 < t_e - t_s < 20:
                                t_start = t_s
                                t_end = t_e
                pdb.set_trace()
            b_spans = torch.nonzero(torch.gt(torch.full(([input_ids.size()[1]]), 99).to(
                input_ids.device), input_ids[pi])).squeeze(0).squeeze(1).detach().tolist()
            token_len = input_ids[pi].nonzero().size()[0]
            b_spans = [i for i in b_spans if i <= token_len-1]
            assert len(b_spans) >= 4
            for i in [h_start, h_end, t_start, t_end]:
                b_spans.remove(i)
            h_span = [h_pos for h_pos in range(h_start, h_end+1)]
            t_span = [t_pos for t_pos in range(t_start, t_end+1)]
            h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(
                0, torch.tensor(h_span).to(input_ids.device), 1)
            t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(
                0, torch.tensor(t_span).to(input_ids.device), 1)
        except: 
            # pdb.set_trace()
            h_span = []
            t_span = []
            h_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device)
            t_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device)
            b_spans = []
        b_span_ = []
        if len(b_spans) > 0 and len(b_spans) % 2 == 0:
            b_span_chunks = [b_spans[i:i+2] for i in range(0, len(b_spans), 2)]
            b_span = []
            for span in b_span_chunks:
                b_span.extend([b_pos for b_pos in range(span[0], span[1]+1)])
            b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(
                0, torch.tensor(b_span).to(input_ids.device), 1)
            b_span_.extend(b_span)
        elif len(b_spans) > 0 and len(b_spans) % 2 == 1:
            b_span = []
            ptr = 0
            # pdb.set_trace()
            while (ptr <= len(b_spans)-1):
                try:
                    if input_ids[pi][b_spans[ptr+1]] - input_ids[pi][b_spans[ptr]] == 1:
                        b_span.append([b_spans[ptr], b_spans[ptr+1]])
                        ptr += 2
                    else:
                        ptr += 1
                except IndexError as e:  
                    # pdb.set_trace()
                    ptr += 1 
            for bs in b_span:
                # pdb.set_trace()
                # ex_bs = range(bs[0], bs[1])
                b_span_.extend(bs)
                if len(b_span_) % 2 != 0:
                    print(b_spans)
            b_span_chunks = [b_span_[i:i+2] for i in range(0, len(b_span_), 2)]
            b_mask = torch.zeros_like(input_ids[pi]).to(input_ids.device).scatter(
                0, torch.tensor(b_span_).to(input_ids.device), 1)
        else:
            b_span_ = []
            b_span_chunks = []
            b_mask = torch.zeros_like(input_ids[pi])
        htb_mask = torch.concat([h_mask.unsqueeze(0), t_mask.unsqueeze(
            0), b_mask.unsqueeze(0)], dim=0)  # [3,512]
        htb_mask_list.append(htb_mask)
        htb_list_batch.append([h_span, t_span, b_span_chunks])
    bag_len = input_ids.size()[0]
    for dp in range(0, bag_len):
        try:
            h_span = htb_list_batch[dp][0]  # [3,4,5,6,7]
            t_span = htb_list_batch[dp][1]  # [71,72,73,74]
            if h_span == [] or t_span == []:
                # print("fail detecting h/t")
                # h_t_flag = False
                # pdb.set_trace()
                # flag do not change
                pass
            else:
                # pdb.set_trace()
                h_t_flag = True
                # print("H/T Detected")
        except Exception as e:
            print(e)
            pdb.set_trace()
    # entity_mask = T[8,3,512] h_mask = T[8, 512] t_mask = T[8,512] b_mask = T[8,512]
    return h_t_flag


def complete_h_t(all_buf, filtered_buf):
    h_markers = [1, 2]
    t_markers = [3, 4]
    for blk_id, blk in enumerate(filtered_buf.blocks):
        if blk.h_flag == 1 and list(set(blk.ids).intersection(set(h_markers))) != h_markers:
            if list(set(blk.ids).intersection(set(h_markers))) == [1]:  
                # pdb.set_trace()
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos].ids
                marker_p = complementary.index(2)
                if 101 in complementary:
                    complementary.remove(101)
                    complementary = complementary[:marker_p]
                else:
                    complementary = complementary[:marker_p+1]
                new = blk.ids + complementary
                filtered_buf[blk_id].ids = new[-len(old):] + [102]
                print(filtered_buf[blk_id].ids)
            elif list(set(blk.ids).intersection(set(h_markers))) == [2]:  
                # pdb.set_trace()
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos-2].ids
                marker_p_start = complementary.index(1)
                if blk.ids[0] != 101:
                    try:
                        marker_p_end = complementary.index(blk.ids[0])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        if complementary[-2] == 1:
                            complementary = [1]
                        else:
                            complementary = [1]
                else:
                    try:
                        marker_p_end = complementary.index(blk.ids[1])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        # pdb.set_trace()
                        if complementary[-2] == 1:
                            complementary = [1]
                        else:
                            complementary = [1]
                if blk.ids[0] != 101:
                    new = complementary + blk.ids
                else:
                    blk.ids.remove(101)
                    new = [101] + complementary + blk.ids
                filtered_buf[blk_id].ids = new[:len(old)] + [102]
                print(filtered_buf[blk_id].ids)
        
        elif blk.h_flag == 1 and list(set(blk.ids).intersection(set(h_markers))) == h_markers:
            pdb.set_trace()
            markers_starts = []
            markers_ends = []
            for i, id in enumerate(blk.ids):
                if id == 1:
                    markers_starts.append(i)
                elif id == 2:
                    markers_ends.append[i]
                else:
                    continue
            if len(markers_starts) > len(markers_ends):  
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos].ids
                marker_p = complementary.index(2)
                if 101 in complementary:
                    complementary.remove(101)
                    complementary = complementary[:marker_p]
                else:
                    complementary = complementary[:marker_p+1]
                new = blk.ids + complementary
                filtered_buf[blk_id].ids = new[-len(old):] + [102]
                print(filtered_buf[blk_id].ids)
            elif len(markers_starts) < len(markers_ends):  
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos-2].ids
                marker_p_start = complementary.index(1)
                if blk.ids[0] != 101:
                    try:
                        marker_p_end = complementary.index(blk.ids[0])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        if complementary[-2] == 1:
                            complementary = [1]
                        else:
                            complementary = [1]
                else:
                    try:
                        marker_p_end = complementary.index(blk.ids[1])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        # pdb.set_trace()
                        if complementary[-2] == 1:
                            complementary = [1]
                        else:
                            complementary = [1]
                if blk.ids[0] != 101:
                    new = complementary + blk.ids
                else:
                    blk.ids.remove(101)
                    new = [101] + complementary + blk.ids
                filtered_buf[blk_id].ids = new[:len(old)] + [102]
                print(filtered_buf[blk_id].ids)

            else:
                if blk.ids.index(2) > blk.ids.index(1):
                    pass
                elif blk.ids.index(2) < blk.ids.index(1):
                    # pdb.set_trace()
                    first_end_marker = blk.ids.index(2)
                    second_start_marker = blk.ids.index(1)
                    
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(2)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)

        elif blk.t_flag == 1 and list(set(blk.ids).intersection(set(t_markers))) != t_markers:
            if list(set(blk.ids).intersection(set(t_markers))) == [3]:  
                # pdb.set_trace()
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos].ids
                marker_p = complementary.index(4)
                if 101 in complementary:
                    complementary.remove(101)
                    complementary = complementary[:marker_p]
                else:
                    complementary = complementary[:marker_p+1]
                new = blk.ids + complementary
                filtered_buf[blk_id].ids = new[-len(old):] + [102]
                print(filtered_buf[blk_id].ids)
            elif list(set(blk.ids).intersection(set(t_markers))) == [4]:  
                # pdb.set_trace()
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos-2].ids
                marker_p = complementary.index(3)
                marker_p_start = complementary.index(3)
                if blk.ids[0] != 101:
                    try:
                        marker_p_end = complementary.index(blk.ids[0])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]

                    except Exception as e:
                        if complementary[-2] == 3:
                            complementary = [3]
                        else:
                            complementary = [3]
                else:
                    try:
                        marker_p_end = complementary.index(blk.ids[1])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        # pdb.set_trace()
                        if complementary[-2] == 3:
                            complementary = [3]
                        else:
                            complementary = [3]
                if blk.ids[0] != 101:
                    new = complementary + blk.ids
                else:
                    blk.ids.remove(101)
                    new = [101] + complementary + blk.ids
                filtered_buf[blk_id].ids = new[:len(old)] + [102]
                print(filtered_buf[blk_id].ids)

        
        elif blk.t_flag == 1 and list(set(blk.ids).intersection(set(t_markers))) == t_markers:
            pdb.set_trace()
            markers_starts = []
            markers_ends = []
            for i, id in enumerate(blk.ids):
                if id == 3:
                    markers_starts.append(i)
                elif id == 4:
                    markers_ends.append[i]
                else:
                    continue
            if len(markers_starts) > len(markers_ends):  
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos].ids
                marker_p = complementary.index(4)
                if 101 in complementary:
                    complementary.remove(101)
                    complementary = complementary[:marker_p]
                else:
                    complementary = complementary[:marker_p+1]
                new = blk.ids + complementary
                filtered_buf[blk_id].ids = new[-len(old):] + [102]
                print(filtered_buf[blk_id].ids)
            elif len(markers_starts) < len(markers_ends):  
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos-2].ids
                marker_p_start = complementary.index(3)
                if blk.ids[0] != 101:
                    try:
                        marker_p_end = complementary.index(blk.ids[0])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        if complementary[-2] == 3:
                            complementary = [3]
                        else:
                            complementary = [3]
                else:
                    try:
                        marker_p_end = complementary.index(blk.ids[1])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        # pdb.set_trace()
                        if complementary[-2] == 3:
                            complementary = [3]
                        else:
                            complementary = [3]
                if blk.ids[0] != 101:
                    new = complementary + blk.ids
                else:
                    blk.ids.remove(101)
                    new = [101] + complementary + blk.ids
                filtered_buf[blk_id].ids = new[:len(old)] + [102]
                print(filtered_buf[blk_id].ids)
            else:  
                if blk.ids.index(4) > blk.ids.index(3):
                    pass
                elif blk.ids.index(4) < blk.ids.index(3):
                    # pdb.set_trace()
                    first_end_marker = blk.ids.index(4)
                    second_start_marker = blk.ids.index(3)
                    
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(4)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
    return filtered_buf


def complete_h_t_debug(all_buf, filtered_buf):
    h_markers = [1, 2]
    t_markers = [3, 4]
    pdb.set_trace()
    for blk_id, blk in enumerate(filtered_buf.blocks):
        if blk.h_flag == 1 and list(set(blk.ids).intersection(set(h_markers))) != h_markers:
            if list(set(blk.ids).intersection(set(h_markers))) == [1]:  
                # pdb.set_trace()
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos].ids
                marker_p = complementary.index(2)
                if 101 in complementary:
                    complementary.remove(101)
                    complementary = complementary[:marker_p]
                else:
                    complementary = complementary[:marker_p+1]
                new = blk.ids + complementary
                filtered_buf[blk_id].ids = new[-len(old):] + [102]
                print(filtered_buf[blk_id].ids)
            elif list(set(blk.ids).intersection(set(h_markers))) == [2]:  
                # pdb.set_trace()
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos-2].ids
                marker_p_start = complementary.index(1)
                if blk.ids[0] != 101:
                    try:
                        marker_p_end = complementary.index(blk.ids[0])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        if complementary[-2] == 1:
                            complementary = [1]
                        else:
                            complementary = [1]
                else:
                    try:
                        marker_p_end = complementary.index(blk.ids[1])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        # pdb.set_trace()
                        if complementary[-2] == 1:
                            complementary = [1]
                        else:
                            complementary = [1]
                if blk.ids[0] != 101:
                    new = complementary + blk.ids
                else:
                    blk.ids.remove(101)
                    new = [101] + complementary + blk.ids
                filtered_buf[blk_id].ids = new[:len(old)] + [102]
                print(filtered_buf[blk_id].ids)
        
        elif blk.h_flag == 1 and list(set(blk.ids).intersection(set(h_markers))) == h_markers:
            # pdb.set_trace()
            markers_starts = []
            markers_ends = []
            for i, id in enumerate(blk.ids):
                if id == 1:
                    markers_starts.append(i)
                elif id == 2:
                    markers_ends.append(i)
                else:
                    continue
            if len(markers_starts) > len(markers_ends):  
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos].ids
                marker_p = complementary.index(2)
                if 101 in complementary:
                    complementary.remove(101)
                    complementary = complementary[:marker_p]
                else:
                    complementary = complementary[:marker_p+1]
                new = blk.ids + complementary
                filtered_buf[blk_id].ids = new[-len(old):] + [102]
                print(filtered_buf[blk_id].ids)
            elif len(markers_starts) < len(markers_ends):  
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos-2].ids
                marker_p_start = complementary.index(1)
                if blk.ids[0] != 101 and blk.ids[0] != 2:
                    try:
                        marker_p_end = complementary.index(blk.ids[0])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        if complementary[-2] == 1:
                            complementary = [1]
                        else:
                            complementary = [1]
                elif blk.ids[0] != 2:
                    try:
                        marker_p_end = complementary.index(blk.ids[1])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        # pdb.set_trace()
                        if complementary[-2] == 1:
                            complementary = [1]
                        else:
                            complementary = [1]
                else:
                    complementary = all_buf[blk.pos-2].ids[marker_p_start:-1]
                if blk.ids[0] != 101:
                    new = complementary + blk.ids
                else:
                    blk.ids.remove(101)
                    new = [101] + complementary + blk.ids
                filtered_buf[blk_id].ids = new[:len(old)] + [102]
                print(filtered_buf[blk_id].ids)

            else:
                if blk.ids.index(2) > blk.ids.index(1):
                    pass
                elif blk.ids.index(2) < blk.ids.index(1):
                    # pdb.set_trace()
                    first_end_marker = blk.ids.index(2)
                    second_start_marker = blk.ids.index(1)
                    
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(2)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)

        elif blk.t_flag == 1 and list(set(blk.ids).intersection(set(t_markers))) != t_markers:
            if list(set(blk.ids).intersection(set(t_markers))) == [3]:  
                # pdb.set_trace()
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos].ids
                marker_p = complementary.index(4)
                if 101 in complementary:
                    complementary.remove(101)
                    complementary = complementary[:marker_p]
                else:
                    complementary = complementary[:marker_p+1]
                new = blk.ids + complementary
                filtered_buf[blk_id].ids = new[-len(old):] + [102]
                print(filtered_buf[blk_id].ids)
            elif list(set(blk.ids).intersection(set(t_markers))) == [4]:  
                # pdb.set_trace()
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos-2].ids
                marker_p = complementary.index(3)
                marker_p_start = complementary.index(3)
                if blk.ids[0] != 101:
                    try:
                        marker_p_end = complementary.index(blk.ids[0])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]

                    except Exception as e:
                        if complementary[-2] == 3:
                            complementary = [3]
                        else:
                            complementary = [3]
                else:
                    try:
                        marker_p_end = complementary.index(blk.ids[1])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        # pdb.set_trace()
                        if complementary[-2] == 3:
                            complementary = [3]
                        else:
                            complementary = [3]
                if blk.ids[0] != 101:
                    new = complementary + blk.ids
                else:
                    blk.ids.remove(101)
                    new = [101] + complementary + blk.ids
                filtered_buf[blk_id].ids = new[:len(old)] + [102]
                print(filtered_buf[blk_id].ids)

        
        elif blk.t_flag == 1 and list(set(blk.ids).intersection(set(t_markers))) == t_markers:
            # pdb.set_trace()
            markers_starts = []
            markers_ends = []
            for i, id in enumerate(blk.ids):
                if id == 3:
                    markers_starts.append(i)
                elif id == 4:
                    markers_ends.append(i)
                else:
                    continue
            if len(markers_starts) > len(markers_ends):  
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos].ids
                marker_p = complementary.index(4)
                if 101 in complementary:
                    complementary.remove(101)
                    complementary = complementary[:marker_p]
                else:
                    complementary = complementary[:marker_p+1]
                new = blk.ids + complementary
                filtered_buf[blk_id].ids = new[-len(old):] + [102]
                print(filtered_buf[blk_id].ids)
            elif len(markers_starts) < len(markers_ends):  
                old = blk.ids
                blk.ids.pop()
                complementary = all_buf[blk.pos-2].ids
                marker_p_start = complementary.index(3)
                if blk.ids[0] != 101 and blk.ids[0] != 4:
                    try:
                        marker_p_end = complementary.index(blk.ids[0])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        if complementary[-2] == 3:
                            complementary = [3]
                        else:
                            complementary = [3]
                elif blk.ids[0] != 4:
                    try:
                        marker_p_end = complementary.index(blk.ids[1])
                        if marker_p_end > marker_p_start:
                            complementary = complementary[marker_p_start:marker_p_end]
                        else:
                            if complementary[-1] == 102:
                                complementary = complementary[marker_p_start:-1]
                            else:
                                complementary = complementary[marker_p_start:]
                    except Exception as e:
                        # pdb.set_trace()
                        if complementary[-2] == 3:
                            complementary = [3]
                        else:
                            complementary = [3]
                else:
                    complementary = all_buf[blk.pos-2].ids[marker_p_start:-1]
                if blk.ids[0] != 101:
                    new = complementary + blk.ids
                else:
                    blk.ids.remove(101)
                    new = [101] + complementary + blk.ids
                filtered_buf[blk_id].ids = new[:len(old)] + [102]
                print(filtered_buf[blk_id].ids)
            else: 
                if blk.ids.index(4) > blk.ids.index(3):
                    pass
                elif blk.ids.index(4) < blk.ids.index(3):
                    # pdb.set_trace()
                    first_end_marker = blk.ids.index(4)
                    second_start_marker = blk.ids.index(3)
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(4)
                    if 101 in complementary:
                        complementary.remove(101)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new[-len(old):] + [102]
                    print(filtered_buf[blk_id].ids)
    return filtered_buf


def if_h_t_complete(buffer):
    h_flag = False
    t_flag = False
    h_markers = [1, 2]
    t_markers = [3, 4]
    for ret in buffer:
        if list(set(ret.ids) & (set(h_markers))) != h_markers:
            continue
        else:
            if ret.ids.index(1) < ret.ids.index(2):
                h_flag = True
            else:
                continue
    for ret in buffer:
        if list(set(ret.ids) & (set(t_markers))) != t_markers:
            continue
        else:
            if ret.ids.index(3) < ret.ids.index(4):
                t_flag = True
            else:
                continue
    if h_flag and t_flag:
        return True
    else:
        return False


def bridge_entity_based_filter(tokenizer, doc0, doc1, num_max_tokens):
    
    alpha = 1
    beta = 0.1
    # gamma = 0.01
    K = num_max_tokens // BLOCK_SIZE + 4
    # print(f'{K = }')

    def complete_h_t(all_buf, filtered_buf):
        h_markers = [1, 2]
        t_markers = [3, 4]
        for blk_id, blk in enumerate(filtered_buf.blocks):

            if blk.h_flag == 1 and not set(h_markers).issubset(blk.ids): 

                if set(blk.ids) & set(h_markers) == { H_START_MARKER_ID }: 
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(H_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p + 1]
                    new = blk.ids + complementary
                    filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    
                elif set(blk.ids) & set(h_markers) == { H_END_MARKER_ID }:
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos - 2].ids
                    marker_p_start = complementary.index(H_START_MARKER_ID)
                    if blk.ids[0] != CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            # pdb.set_trace()
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                    
            elif blk.h_flag == 1 and list(set(blk.ids) & (set(h_markers))) == h_markers:
                # pdb.set_trace()
                markers_starts = []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == H_START_MARKER_ID:
                        markers_starts.append(i)
                    elif id == H_END_MARKER_ID:
                        markers_ends.append(i)
                    else:
                        continue
                if len(markers_starts) > len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(H_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    if len(new) <= 63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(H_START_MARKER_ID)
                    if blk.ids[0] != CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            # pdb.set_trace()
                            if complementary[-2] == H_START_MARKER_ID:
                                complementary = [H_START_MARKER_ID]
                            else:
                                complementary = [H_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)

                else:
                    if blk.ids.index(H_END_MARKER_ID) < blk.ids.index(H_START_MARKER_ID):
                        # first_end_marker = blk.ids.index(H_END_MARKER_ID)
                        # second_start_marker = blk.ids.index(H_START_MARKER_ID)
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(H_END_MARKER_ID)
                        if CLS_TOKEN_ID in complementary:
                            complementary.remove(CLS_TOKEN_ID)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p+1]
                        new = blk.ids + complementary
                        if len(new) <= 63:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        else:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        # print(filtered_buf[blk_id].ids)

            elif blk.t_flag == 1 and list(set(blk.ids) & (set(t_markers))) != t_markers:
                if list(set(blk.ids) & (set(t_markers))) == [T_START_MARKER_ID]:
                    # pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(T_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    if len(new) <= 63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
                elif list(set(blk.ids) & (set(t_markers))) == [T_END_MARKER_ID]:
                    # pdb.set_trace()
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p = complementary.index(T_START_MARKER_ID)
                    marker_p_start = complementary.index(T_START_MARKER_ID)
                    if blk.ids[0] != CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]

                        except Exception as e:
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            # pdb.set_trace()
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)

            elif blk.t_flag == 1 and list(set(blk.ids) & (set(t_markers))) == t_markers:
                # pdb.set_trace()
                markers_starts = []
                markers_ends = []
                for i, id in enumerate(blk.ids):
                    if id == T_START_MARKER_ID:
                        markers_starts.append(i)
                    elif id == T_END_MARKER_ID:
                        markers_ends.append(i)
                    else:
                        continue
                if len(markers_starts) > len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos].ids
                    marker_p = complementary.index(T_END_MARKER_ID)
                    if CLS_TOKEN_ID in complementary:
                        complementary.remove(CLS_TOKEN_ID)
                        complementary = complementary[:marker_p]
                    else:
                        complementary = complementary[:marker_p+1]
                    new = blk.ids + complementary
                    if len(new) <= 63:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    else:
                        filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                    # print(filtered_buf[blk_id].ids)
                elif len(markers_starts) < len(markers_ends):
                    old = blk.ids
                    blk.ids.pop()
                    complementary = all_buf[blk.pos-2].ids
                    marker_p_start = complementary.index(3)
                    if blk.ids[0] != CLS_TOKEN_ID:
                        try:
                            marker_p_end = complementary.index(blk.ids[0])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    else:
                        try:
                            marker_p_end = complementary.index(blk.ids[1])
                            if marker_p_end > marker_p_start:
                                complementary = complementary[marker_p_start:marker_p_end]
                            else:
                                if complementary[-1] == SEP_TOKEN_ID:
                                    complementary = complementary[marker_p_start:-1]
                                else:
                                    complementary = complementary[marker_p_start:]
                        except Exception as e:
                            # pdb.set_trace()
                            if complementary[-2] == T_START_MARKER_ID:
                                complementary = [T_START_MARKER_ID]
                            else:
                                complementary = [T_START_MARKER_ID]
                    if blk.ids[0] != CLS_TOKEN_ID:
                        new = complementary + blk.ids
                    else:
                        blk.ids.remove(CLS_TOKEN_ID)
                        new = [CLS_TOKEN_ID] + complementary + blk.ids
                    filtered_buf[blk_id].ids = new[:len(old)] + [SEP_TOKEN_ID]
                else:
                    if blk.ids.index(T_END_MARKER_ID) < blk.ids.index(T_START_MARKER_ID):
                        # first_end_marker = blk.ids.index(T_END_MARKER_ID)
                        # second_start_marker = blk.ids.index(T_START_MARKER_ID)
                        old = blk.ids
                        blk.ids.pop()
                        complementary = all_buf[blk.pos].ids
                        marker_p = complementary.index(T_END_MARKER_ID)
                        if CLS_TOKEN_ID in complementary:
                            complementary.remove(CLS_TOKEN_ID)
                            complementary = complementary[:marker_p]
                        else:
                            complementary = complementary[:marker_p+1]
                        new = blk.ids + complementary
                        if len(new) <= 63:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        else:
                            filtered_buf[blk_id].ids = new + [SEP_TOKEN_ID]
                        if filtered_buf[blk_id].ids[0] != CLS_TOKEN_ID:
                            filtered_buf[blk_id].ids = [
                                CLS_TOKEN_ID] + filtered_buf[blk_id].ids[1:]
                        else:
                            continue
                        # print(filtered_buf[blk_id].ids)
            
            if filtered_buf[blk_id].ids[0] != CLS_TOKEN_ID and filtered_buf[blk_id].ids[0] not in [1, 3]:
                if len(filtered_buf[blk_id].ids) <= 63:
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[:]
                else:
                    # pdb.set_trace()
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[1:]
            
            elif filtered_buf[blk_id].ids[0] in [1, 3]:
                if len(filtered_buf[blk_id].ids) <= 63:
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[:]
                else:
                    filtered_buf[blk_id].ids = [CLS_TOKEN_ID] + filtered_buf[blk_id].ids[:]
                    # pdb.set_trace()
            
            if filtered_buf[blk_id].ids[0] != CLS_TOKEN_ID or filtered_buf[blk_id].ids[-1] != SEP_TOKEN_ID:
                pdb.set_trace()

        return filtered_buf

    def detect_h_t(tokenizer, buffer):
        h_markers = ["[unused" + str(i) + "]" for i in range(1, 3)]
        t_markers = ["[unused" + str(i) + "]" for i in range(3, 5)]
        h_blocks = []
        t_blocks = []
        for blk in buffer:
            if list(set(tokenizer.convert_tokens_to_ids(h_markers)) & (set(blk.ids))):
                h_blocks.append(blk)
            elif list(set(tokenizer.convert_tokens_to_ids(t_markers)) & (set(blk.ids))):
                t_blocks.append(blk)
            else:
                continue
        return h_blocks, t_blocks

    def if_h_t_complete(buffer):
        h_flag = False
        t_flag = False
        h_markers = [1, 2]
        t_markers = [3, 4]
        for ret in buffer:
            if list(set(ret.ids) & (set(h_markers))) != h_markers:
                continue
            else:
                if ret.ids.index(1) < ret.ids.index(2):
                    h_flag = True
                else:
                    if len(list(set(ret.ids) & (set([2])))) > len(list(set(ret.ids) & (set([1])))):
                        h_flag = True
                    else:
                        continue
        for ret in buffer:
            if list(set(ret.ids) & (set(t_markers))) != t_markers:
                continue
            else:
                if ret.ids.index(3) < ret.ids.index(T_END_MARKER_ID):
                    t_flag = True
                else:
                    if len(list(set(ret.ids) & (set([T_END_MARKER_ID])))) > len(list(set(ret.ids) & (set([3])))):
                        t_flag = True
                    else:
                        continue
        if h_flag and t_flag:
            return True
        else:
            return False

    def co_occur_graph(tokenizer, d0, d1, alpha, beta):
        h_markers = ["[unused" + str(i) + "]" for i in range(1, 3)]
        t_markers = ["[unused" + str(i) + "]" for i in range(3, 5)]
        ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
        b_markers  = ["[unused" + str(i) + "]" for i in range(5, 101)]
        # max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        cnt, batches = 0, []
        d = []

        for di in [d0, d1]:
            d.extend(di)
        d0_buf, cnt = Buffer.split_document_into_blocks(d0, tokenizer, num_max_tokens, cnt=cnt, hard=False, docid=0)
        d1_buf, cnt = Buffer.split_document_into_blocks(d1, tokenizer, num_max_tokens, cnt=cnt, hard=False, docid=1)
        dbuf = Buffer(num_max_tokens)
        dbuf.blocks = d0_buf.blocks + d1_buf.blocks
        for blk in dbuf.blocks:
            if blk.ids[0] != CLS_TOKEN_ID:
                blk.ids = [CLS_TOKEN_ID] + blk.ids

        co_occur_pair = []
        for blk in dbuf:
            if set(tokenizer.convert_tokens_to_ids(h_markers)) & set(blk.ids) and set(tokenizer.convert_tokens_to_ids(b_markers)) & set(blk.ids):
                b_idx = list(set([math.ceil(int(b_m)/2) for b_m in list(
                    set(tokenizer.convert_tokens_to_ids(b_markers)) & (set(blk.ids)))]))[0]
                co_occur_pair.append((1, b_idx, blk.pos))
            elif set(tokenizer.convert_tokens_to_ids(t_markers)) & set(blk.ids) and set(tokenizer.convert_tokens_to_ids(b_markers)) & set(blk.ids):
                b_idx = list(set([math.ceil(int(b_m)/2) for b_m in list(
                    set(tokenizer.convert_tokens_to_ids(b_markers)) & (set(blk.ids)))]))[0]
                co_occur_pair.append((2, b_idx, blk.pos))
            elif set(tokenizer.convert_tokens_to_ids(b_markers)) & set(blk.ids):
                b_idxs = { 
                    math.ceil(int(b_m) / 2) 
                    for b_m in set(tokenizer.convert_tokens_to_ids(b_markers)) & set(blk.ids)
                }
                if len(b_idxs) >= 2:
                    pairs = combinations(b_idxs, 2)
                else:
                    pairs = []
                for pair in pairs:
                    co_occur_pair.append((pair[0], pair[1], blk.pos))

        h_co = list(filter(lambda pair: pair[0] == 1, co_occur_pair))
        t_co = list(filter(lambda pair: pair[0] == 2, co_occur_pair))
        b_co = list(filter(lambda pair: pair[0] >  2, co_occur_pair))

        score_b = {}
        s1 = {}
        s2 = {}
        # s3 = {}

        for entity_id in range(1, math.ceil(len(b_markers) / 2) + 2):
            s1[entity_id] = 0
            s2[entity_id] = 0
            # s3[entity_id] = 0
            score_b[entity_id] = 0

        for pair in co_occur_pair:
            if pair[0] <= 2:
                s1[pair[1]] = 1

        for pair in b_co:
            if s1[pair[0]] == 1:
                s2[pair[1]] += 1
            if s1[pair[1]] == 1:
                s2[pair[0]] += 1

        # bridge_ids = {doc_entities[dps_count][key]: key for key in doc_entities[dps_count].keys()}
        # for idx in range(len(doc_entities)):
        #     if idx == dps_count:
        #         continue
        #     ent_ids = doc_entities[idx].keys()
        #     for k, v in bridge_ids.items():
        #         if v in ent_ids:
        #             s3[k+3] += 1

        for entity_id in range(1, math.ceil(len(b_markers) / 2) + 2):
            score_b[entity_id] += alpha * s1[entity_id] + beta * s2[entity_id]  # + gamma * s3[entity_id]
        # pdb.set_trace()
        return score_b

    def get_block_by_sentence_score(tokenizer, d0, d1, score_b, K):
        # pdb.set_trace()
        h_markers = ["[unused" + str(i) + "]" for i in range(1, 3)]
        t_markers = ["[unused" + str(i) + "]" for i in range(3, 5)]
        ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
        b_markers = ["[unused" + str(i) + "]" for i in range(5, 101)]
        # max_blk_num = CAPACITY // (BLOCK_SIZE + 1)
        cnt, batches = 0, []

        score_b_positive = [(k, v) for k, v in score_b.items() if v > 0]
        score_b_positive_ids = []
        for b in score_b_positive:
            b_id = b[0]
            b_score = b[1]
            score_b_positive_ids.append(2 * b_id - 1)
            score_b_positive_ids.append(2 * b_id)

        # pdb.set_trace()
        d0_buf, cnt = Buffer.split_document_into_blocks(d0, tokenizer, num_max_tokens, cnt=cnt, hard=False, docid=0)
        d1_buf, cnt = Buffer.split_document_into_blocks(d1, tokenizer, num_max_tokens, cnt=cnt, hard=False, docid=1)
        dbuf_all = Buffer(num_max_tokens)
        dbuf_all.blocks = d0_buf.blocks + d1_buf.blocks
        for blk in dbuf_all.blocks:
            if blk.ids[0] != CLS_TOKEN_ID:
                blk.ids = [CLS_TOKEN_ID] + blk.ids
            if set(tokenizer.convert_tokens_to_ids(h_markers)) & set(blk.ids):
                blk.h_flag = 1
            elif set(tokenizer.convert_tokens_to_ids(t_markers)) & set(blk.ids):
                blk.t_flag = 1

        for blk in dbuf_all:
            if set(score_b_positive_ids) & set(blk.ids):
                blk_bridge_marker_ids = set(score_b_positive_ids) & set(blk.ids)
                blk_bridge_ids = { math.ceil(int(b_m_id) / 2) for b_m_id in blk_bridge_marker_ids }
                for b_id in blk_bridge_ids:
                    blk.relevance += score_b[b_id]
                # print(blk.pos, blk.relevance)
            else:
                blk.relevance = 0

        # pdb.set_trace()
        for blk in dbuf_all:
            if blk.h_flag == 1 or blk.t_flag == 1:
                blk.relevance += 1

        block_scores = {}
        for blk in dbuf_all:
            block_scores[blk.pos] = blk.relevance
            # print(blk.pos, blk.relevance)

        block_scores = sorted(block_scores.items(), key=lambda x: x[1], reverse=True)

        try:
            score_threshold = block_scores[K][1]
        except IndexError as e:
            h_blocks = []
            t_blocks = []
            if not if_h_t_complete(dbuf_all):
                # pdb.set_trace()
                dbuf_all = complete_h_t(dbuf_all, dbuf_all)
                if not if_h_t_complete(dbuf_all):
                    pdb.set_trace()
                    dbuf_all = complete_h_t_debug(dbuf_all, dbuf_all)

            for blk in dbuf_all:
                if set(tokenizer.convert_tokens_to_ids(h_markers)) & set(blk.ids):
                    h_blocks.append(blk)
                elif set(tokenizer.convert_tokens_to_ids(t_markers)) & set(blk.ids):
                    t_blocks.append(blk)
            
            return h_blocks, t_blocks, dbuf_all, dbuf_all

        score_highest = block_scores[0][1]
        if score_threshold > 0 or score_highest > 0:
            p_buf, n_buf = dbuf_all.filtered(lambda blk, idx: blk.relevance > score_threshold, need_residue=True)
            e_buf, n_buf = dbuf_all.filtered(lambda blk, idx: blk.relevance == score_threshold, need_residue=True)
        else:
            p_buf, e_buf = dbuf_all.filtered(lambda blk, idx: blk.h_flag + blk.t_flag > 0, need_residue=True)

        if len(p_buf) + len(e_buf) == K:
            dbuf_filtered = p_buf + e_buf
        elif len(p_buf) + len(e_buf) < K:
            _, rest_buf = dbuf_all.filtered(lambda blk, idx: blk.relevance < score_threshold, need_residue=True)
            dbuf_filtered = p_buf + e_buf + random.sample(rest_buf, K - len(p_buf) - len(e_buf))
            assert len(dbuf_filtered) <= K
        else:
            try:
                highest_blk_id = sorted(p_buf, key=lambda x: x.relevance, reverse=True)[0].pos
            except:
                if score_threshold > 0 or score_highest > 0:
                    highest_blk_id = sorted(e_buf, key=lambda x: x.relevance, reverse=True)[0].pos
                else:
                    detect_h_t(tokenizer, dbuf_filtered)
            e_buf_selected_blocks = []
            try:
                if sorted(p_buf, key=lambda x: x.relevance, reverse=True)[0].relevance > 0:
                    e_buf_distance = {}
                    for idx, e in enumerate(e_buf):
                        e_buf_distance[idx] = abs(e.pos - highest_blk_id)
                    e_buf_distance = sorted(
                        e_buf_distance.items(), key=lambda x: x[1], reverse=False)
                    e_buf_selected = [k_d[0]
                                      for k_d in e_buf_distance[:K-len(p_buf)]]
                    for e_b_s in e_buf_selected:
                        e_buf_selected_blocks.append(e_buf[e_b_s])
            except:
                if e_buf[0].relevance > 0:
                    e_buf_distance = {}
                    ht_buf, _ = dbuf_all.filtered(lambda blk, idx: blk.h_flag + blk.t_flag > 0, need_residue=True)
                    for idx, e in enumerate(e_buf):
                        e_buf_distance[idx] = min([abs(e.pos - ht_blk.pos) for ht_blk in ht_buf.blocks])
                    e_buf_distance = sorted(e_buf_distance.items(), key=lambda x: x[1], reverse=False)
                    e_buf_selected = [k_d[0] for k_d in e_buf_distance[:K-len(p_buf)]]
                    for e_b_s in e_buf_selected:
                        e_buf_selected_blocks.append(e_buf[e_b_s])
                else:
                    e_buf_distance = {}
                    for idx, e in enumerate(e_buf):
                        e_buf_distance[idx] = min([abs(e.pos - ht_blk.pos) for ht_blk in p_buf.blocks])
                    e_buf_distance = sorted(e_buf_distance.items(), key=lambda x: x[1], reverse=False)
                    e_buf_selected = [k_d[0] for k_d in e_buf_distance[:K-len(p_buf)]]
                    for e_b_s in e_buf_selected:
                        e_buf_selected_blocks.append(e_buf[e_b_s])
            dbuf_blocks = p_buf.blocks + e_buf_selected_blocks
            dbuf_filtered = Buffer(num_max_tokens)
            for block in dbuf_blocks:
                dbuf_filtered.insert(block)

        h_blocks = []
        t_blocks = []
        for blk in dbuf_filtered:
            if set(tokenizer.convert_tokens_to_ids(h_markers)) & set(blk.ids):
                h_blocks.append(blk)
            elif set(tokenizer.convert_tokens_to_ids(t_markers)) & set(blk.ids):
                t_blocks.append(blk)

        if len(h_blocks) == 0 or len(t_blocks) == 0:
            new_dbuf = Buffer(num_max_tokens)
            ori_dbuf_all_blocks = sorted(
                dbuf_all.blocks, 
                key=lambda x: x.relevance * 0.01 + (x.h_flag + x.t_flag), 
                reverse=True
            )
            ori_dbuf_filtered_blocks = sorted(
                dbuf_filtered.blocks, 
                key=lambda x: x.relevance * 0.01 + (x.h_flag + x.t_flag), 
                reverse=True
            )
            if len(h_blocks) == 0:
                candi_h_blocks = []
                for blk in ori_dbuf_all_blocks:
                    if blk.h_flag:
                        candi_h_blocks.append(blk)
                h_blocks.append(random.choice(candi_h_blocks))
                new_dbuf.insert(h_blocks[0])
            if len(t_blocks) == 0:
                candi_t_blocks = []
                for blk in ori_dbuf_all_blocks:
                    if blk.t_flag:
                        candi_t_blocks.append(blk)
                t_blocks.append(random.choice(candi_t_blocks))
                new_dbuf.insert(t_blocks[0])
            for ori_blk in ori_dbuf_filtered_blocks:
                if len(new_dbuf) >= K:
                    break
                new_dbuf.insert(ori_blk)
            dbuf_filtered = new_dbuf

        h_t_block_pos = [blk.pos for blk in h_blocks] + [blk.pos for blk in t_blocks]
        all_block_pos = [blk.pos for blk in dbuf_filtered]
        
        if not set(h_t_block_pos).issubset(all_block_pos):
            if len(set(all_block_pos) & set(h_t_block_pos)) < len(h_t_block_pos): 
                h_blocks = [blk for blk in dbuf_filtered if blk.h_flag == 1]
                t_blocks = [blk for blk in dbuf_filtered if blk.t_flag == 1]
                h_t_block_pos = [blk.pos for blk in h_blocks] + [blk.pos for blk in t_blocks]
                assert len(set(all_block_pos) & set(h_t_block_pos)) == len(h_t_block_pos)
            else:
                pdb.set_trace()
        
        if not if_h_t_complete(dbuf_filtered):
            dbuf_filtered = complete_h_t(dbuf_all, dbuf_filtered)
            if not if_h_t_complete(dbuf_filtered):
                pdb.set_trace()
                dbuf_filtered = complete_h_t_debug(dbuf_all, dbuf_filtered)
        
        return h_blocks, t_blocks, dbuf_filtered, dbuf_all

    score_b = co_occur_graph(tokenizer, doc0, doc1, alpha, beta)
    h_blocks, t_blocks, dbuf, dbuf_all = get_block_by_sentence_score(tokenizer, doc0, doc1, score_b, K)
    if len(h_blocks) == 0 or len(t_blocks) == 0:
        pdb.set_trace()
    
    h_t_flag = False
    dbuf_concat = []
    for blk in dbuf:
        dbuf_concat.extend(blk.ids)
    h_t_flag = check_htb(torch.tensor(dbuf_concat).unsqueeze(0), h_t_flag) 
    if not h_t_flag:
        pdb.set_trace()
        h_t_flag = check_htb_debug(torch.tensor(dbuf_concat).unsqueeze(0), h_t_flag)
    
    return h_blocks, t_blocks, dbuf, dbuf_all


def sent_filter(tokenizer, h, t, doc0, doc1, sbert_wk, num_max_tokens):
    def fix_entity(doc, ht_markers, b_markers):
        markers = ht_markers + b_markers
        markers_pos = []
        if list(set(doc) & (set(markers))):
            for marker in markers:
                try:
                    pos = doc.index(marker)
                    markers_pos.append((pos, marker))
                except ValueError as e:
                    continue

        idx = 0
        while idx <= len(markers_pos)-1:
            try:
                assert (int(markers_pos[idx][1].replace("[unused", "").replace("]", "")) % 2 == 1) and (int(markers_pos[idx][1].replace(
                    "[unused", "").replace("]", "")) - int(markers_pos[idx+1][1].replace("[unused", "").replace("]", "")) == -1)
                entity_name = doc[markers_pos[idx]
                                  [0]+1: markers_pos[idx + 1][0]]
                while "." in entity_name:
                    assert doc[markers_pos[idx][0] +
                               entity_name.index(".") + 1] == "."
                    doc[markers_pos[idx][0] + entity_name.index(".") + 1] = "|"
                    entity_name = doc[markers_pos[idx]
                                      [0]+1: markers_pos[idx + 1][0]]
                idx += 2
            except:
                idx += 1
                continue
        return doc

    ht_markers = ["[unused" + str(i) + "]" for i in range(1, 5)]
    b_markers  = ["[unused" + str(i) + "]" for i in range(5, 101)]
    doc0 = fix_entity(doc0, ht_markers, b_markers)
    doc1 = fix_entity(doc1, ht_markers, b_markers)

    # len(dbuf) == K
    h_blocks, t_blocks, dbuf, dbuf_all = bridge_entity_based_filter(tokenizer, doc0, doc1, num_max_tokens)
    # print(f"{len(dbuf) = }")

    sentence_blocks = dbuf.blocks
    block_pos = [blk.pos for blk in dbuf]
    order_start_blocks = [blk.pos for blk in h_blocks]
    order_end_blocks   = [blk.pos for blk in t_blocks]
    if len(order_start_blocks) == 0 or len(order_end_blocks) == 0:
        pdb.set_trace()

    doc_0_sentences = [
        tokenizer.convert_ids_to_tokens(blk.ids) 
        for blk in sentence_blocks
        if blk.docid == 0
    ]
    doc_1_sentences = [
        tokenizer.convert_ids_to_tokens(blk.ids) 
        for blk in sentence_blocks
        if blk.docid == 1
    ]
    for s in doc_0_sentences:
        if '[CLS]' in s:
            s.remove('[CLS]')
        if '[SEP]' in s:
            s.remove('[SEP]')
    for s in doc_1_sentences:
        if '[CLS]' in s:
            s.remove('[CLS]')
        if '[SEP]' in s:
            s.remove('[SEP]')

    try:
        order_starts = [block_pos.index(pos) for pos in order_start_blocks]
        order_ends   = [block_pos.index(pos) for pos in order_end_blocks]
    except:
        pdb.set_trace()
    # pdb.set_trace()
    
    sro = SentReOrdering(
        doc_0_sentences, doc_1_sentences, 
        tokenizer=tokenizer, 
        h=h, t=t, 
        sbert_wk=sbert_wk
    )
    orders = sro.semantic_based_sort(order_starts, order_ends)
    # print(f'{len(orders[0]) = }')

    selected_buffers = []
    for order in orders:
        selected_buffer = Buffer(num_max_tokens)
        cur_total_tokens = sum(
            len(sentence_blocks[od]) for od in order
        )
        # print(f'Before while loop, {cur_total_tokens = }')
        od_scores = {
            od: sentence_blocks[od].relevance
            for od in order[1: -1]
        }
        od_scores = sorted(od_scores.items(), key=lambda s: s[1], reverse=True)
        while cur_total_tokens > num_max_tokens:
            # print(f'{od_scores = }')
            # print(f'{cur_total_tokens = }')
            lowest_score = od_scores[-1][1]
            removable = list(filter(lambda o_score: o_score[1] == lowest_score, od_scores))
            if len(removable) >= 1:
                random.shuffle(removable)
                remove_o = removable[0][0]
            order.remove(remove_o)
            od_scores.remove((remove_o, lowest_score))
            cur_total_tokens -= len(sentence_blocks[remove_o])
        for od in order:
            selected_buffer.insert(sentence_blocks[od])
        selected_buffers.append(selected_buffer)
    # pdb.set_trace()

    return selected_buffers, dbuf_all


def process_example_ReoS(h, t, doc1, doc2, tokenizer, redisd, sbert_wk, num_max_tokens):
    doc1 = json.loads(redisd.get('codred-doc-' + doc1))
    doc2 = json.loads(redisd.get('codred-doc-' + doc2))

    v_h = None
    for entity in doc1['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == h and v_h is None:
            v_h = entity
    assert v_h is not None
    v_t = None
    for entity in doc2['entities']:
        if 'Q' in entity and 'Q' + str(entity['Q']) == t and v_t is None:
            v_t = entity
    assert v_t is not None
    
    d1_v = {}
    for entity in doc1['entities']:
        if 'Q' in entity:
            d1_v[entity['Q']] = entity
    d2_v = {}
    for entity in doc2['entities']:
        if 'Q' in entity:
            d2_v[entity['Q']] = entity
    
    ov = set(d1_v.keys()) & set(d2_v.keys())
    if len(ov) > 40:
        ov = set(random.choices(list(ov), k=40))
    ov = list(ov)
    
    ma = {}
    for e in ov:
        ma[e] = len(ma)

    d1_start = {}
    d1_end = {}
    for entity in doc1['entities']:
        if 'Q' in entity and entity['Q'] in ma:
            for span in entity['spans']:
                d1_start[span[0]] = ma[entity['Q']]
                d1_end[span[1] - 1] = ma[entity['Q']]
    d2_start = {}
    d2_end = {}
    for entity in doc2['entities']:
        if 'Q' in entity and entity['Q'] in ma:
            for span in entity['spans']:
                d2_start[span[0]] = ma[entity['Q']]
                d2_end[span[1] - 1] = ma[entity['Q']]

    k1 = gen_c(
        passage=doc1['tokens'], 
        span=v_h['spans'][0], 
        bound_tokens=['[unused1]', '[unused2]'],
        d_start=d1_start, d_end=d1_end
    )

    k2 = gen_c(
        passage=doc2['tokens'], 
        span=v_t['spans'][0], 
        bound_tokens=['[unused3]', '[unused4]'], 
        d_start=d2_start, d_end=d2_end
    )

    selected_order_rets, dbuf_all = sent_filter(
        tokenizer=tokenizer, 
        h=v_h['name'], t=v_t['name'], 
        doc0=k1, doc1=k2, 
        sbert_wk=sbert_wk, 
        num_max_tokens=num_max_tokens
    )

    if len(selected_order_rets) == 0:
        logging.error("SELECTION FAIL")
        pdb.set_trace()
        return []
    
    h_markers = {1, 2}
    t_markers = {3, 4}
    for selected_order_ret in selected_order_rets: 
        h_flag = False
        t_flag = False
        for sentence in selected_order_ret:
            if h_markers.issubset(sentence.ids): 
                h_flag = True
            if t_markers.issubset(sentence.ids): 
                t_flag = True
            vertical_bar_token_id = tokenizer.convert_tokens_to_ids("|")
            period_token_id = tokenizer.convert_tokens_to_ids(".")
            while vertical_bar_token_id in sentence.ids:
                vertical_bar_idx = sentence.ids.index(vertical_bar_token_id)
                sentence.ids[vertical_bar_idx] = period_token_id

        assert h_flag and t_flag, f'{h_flag = }\t{t_flag = }'

    token_ids, _, _ = selected_order_rets[0].export_01_turn()
    # print(f'{len(token_ids) = }')
    special_tokens_to_filter = {'[PAD]', '[CLS]', '[SEP]', '[MASK]'}  # except [UNK]
    selected_text_path = tokenizer.convert_tokens_to_string(
        [
            token for token in tokenizer.convert_ids_to_tokens(token_ids)
            if token not in special_tokens_to_filter and not token.startswith('[unused')  # skip entity markers and special tokens
        ]
    )
    return selected_text_path, v_h['name'], v_t['name']
