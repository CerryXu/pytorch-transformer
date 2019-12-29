def padding_mask(seq_k, seq_q):
	# seq_k和seq_q的形状都是[B,L]

    len_q = seq_q.size(1)
    
    # `PAD` is 0

    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask