import torch

def weighted_mse(recon, target, weights):
    error = (recon - target) ** 2
    weighted_error = error * weights
    return weighted_error

def train_one_epoch(model, data, optimizer, weights, noise=0.05):
    model.train()
    optimizer.zero_grad()

    target = data["transaction"].x
    # Corrupt data slighltly so model learns patterns
    noisy = target + torch.randn_like(target) * noise

    data["transaction"].x = noisy

    recon, _ = model(data.x_dict, data.edge_index_dict)
    loss = weighted_mse(recon, target, weights).mean()

    loss.backward()
    optimizer.step()
    
    data["transaction"].x = target
    return loss.item()

def evaluate(model, data, weights):
    model.eval()

    with torch.no_grad():
        recon, z_dict = model(data.x_dict, data.edge_index_dict)
    target = data["transaction"].x
    mask = data["transaction"].mask
    
    error = weighted_mse(recon, target, weights)
    loss = error[mask].mean()
    scores = error.mean(dim=1)

    return loss, scores[mask], z_dict

