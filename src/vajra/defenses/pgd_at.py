import torch
from vajra.attacks.pgd import pgd_attack

def adversarial_train_step(model, x, y, optimizer, criterion, device):
    """One step of PGD Adversarial Training."""
    model.train()
    
    # 1. Generate adversarial examples on the fly
    # We use a slightly smaller epsilon during training to balance clean accuracy
    pgd_eps = 0.03
    model.eval()
    result = pgd_attack(model, x, y, epsilon=pgd_eps, num_iter=7)
    x_adv = result.adversarial
    model.train()
    
    # 2. Train on the mixture of clean and adversarial
    optimizer.zero_grad(set_to_none=True)
    
    # Loss on clean batch
    logits_clean = model(x)
    loss_clean = criterion(logits_clean, y)
    
    # Loss on adversarial batch
    logits_adv = model(x_adv)
    loss_adv = criterion(logits_adv, y)
    
    # Total loss (50/50 split)
    loss = 0.5 * loss_clean + 0.5 * loss_adv
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()