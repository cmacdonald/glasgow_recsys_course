import torch
# Helper functions for ranking metrics (HR@k and NDCG@k)
def hit_rate_at_k(ranked_list, ground_truth, k=10):
    return int(ground_truth in ranked_list[:k])

def ndcg_at_k(ranked_list, ground_truth, k=10):
    if ground_truth in ranked_list[:k]:
        index = ranked_list[:k].index(ground_truth)
        return 1 / np.log2(index + 2)
    else:
        return 0

# by sampling negative items (this is a simplified version)
def evaluate_model(model, df, k=10):
    # Check if model has an eval() method
    if hasattr(model, 'eval'):
        model.eval()

    hr_list = []
    ndcg_list = []
    for idx, row in df.iterrows():
        user = torch.tensor([row['userIndex']], dtype=torch.long).to(default_device)
        pos_item = row['movieIndex']
        # Sample 99 negative items (randomly chosen)
        negatives = []
        while len(negatives) < 99:
            neg_item = np.random.randint(0, num_items)
            if neg_item != pos_item:
                negatives.append(neg_item)
        items = [pos_item] + negatives
        items_tensor = torch.tensor(items, dtype=torch.long).to(default_device)
        users_tensor = user.repeat(len(items))
        scores = model(users_tensor, items_tensor).detach().cpu().numpy().tolist()
        # Rank items in descending order
        ranked_items = [x for _, x in sorted(zip(scores, items), key=lambda pair: pair[0], reverse=True)]
        hr = hit_rate_at_k(ranked_items, pos_item, k)
        ndcg = ndcg_at_k(ranked_items, pos_item, k)
        hr_list.append(hr)
        ndcg_list.append(ndcg)
    return np.mean(hr_list), np.mean(ndcg_list)
