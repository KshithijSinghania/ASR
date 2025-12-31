import torch

def ctc_beam_search(log_probs, idx2char, beam_width=5, blank=0):
    """
    Simple CTC beam search decoder.
    log_probs: Tensor of shape (T, C)
    """
    beams = {("", blank): 0.0}

    for t in range(log_probs.size(0)):
        new_beams = {}

        for (prefix, last_char), score in beams.items():
            for c in range(log_probs.size(1)):
                new_score = score + log_probs[t, c].item()

                if c == blank:
                    key = (prefix, last_char)
                else:
                    char = idx2char[c]
                    if c == last_char:
                        key = (prefix, last_char)
                    else:
                        key = (prefix + char, c)

                if key not in new_beams or new_beams[key] < new_score:
                    new_beams[key] = new_score

        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        )

    return max(beams.items(), key=lambda x: x[1])[0][0]
