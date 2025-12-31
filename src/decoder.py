def ctc_greedy_decode(logits):
    preds = logits.argmax(dim=-1)

    decoded = []
    prev = None
    for p in preds:
        if p != prev and p != 0:
            decoded.append(p.item())
        prev = p

    return decoded

def decode_logits(logits, idx2char):
    seq = ctc_greedy_decode(logits)
    return "".join([idx2char[i] for i in seq])
