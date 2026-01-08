import torch


def decode_logits(log_probs, idx2char):
    """
    Greedy CTC decoding

    log_probs: (T, vocab_size)
    """
    pred_ids = torch.argmax(log_probs, dim=-1)

    decoded = []
    prev = None

    for i in pred_ids:
        i = i.item()
        if i != prev and i != 0:
            decoded.append(idx2char[i])
        prev = i

    return "".join(decoded)

def beam_search_decode(log_probs, idx2char, beam_width=10):
    T, V = log_probs.shape
    beams = [(0.0, [])]

    for t in range(T):
        new_beams = []
        for score, seq in beams:
            for v in range(V):
                new_score = score + log_probs[t, v].item()
                new_seq = seq + [v]
                new_beams.append((new_score, new_seq))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

    best_seq = beams[0][1]

    # CTC collapse
    decoded = []
    prev = None
    for i in best_seq:
        if i != prev and i != 0:
            decoded.append(idx2char[i])
        prev = i

    return "".join(decoded)
