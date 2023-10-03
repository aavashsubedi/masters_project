
def hamming_loss(suggested, target):
    errors = suggested * (1.0 - target) + (1.0 - suggested) * target
    return errors.mean(dim=0).sum()