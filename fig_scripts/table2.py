import torch

from tqdm import tqdm
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader

def evaluate(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

batch_size = 32
# Descriptions are available here : https://ogb.stanford.edu/docs/graphprop/
# Try using ogbg-molhiv or ogbg-molpcba
datasets = {'ogbg-ppa': ['mf-ppa', 'ep-ppa'], 
           'cora': ['mf-cora', 'ep-cora']
           }

results = []
for dataset_id, models in datasets.items():

    dataset = PygGraphPropPredDataset(name=dataset_id)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False, num_workers=10)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False, num_workers=10)


    for model_id in models:
        model = load_model(model)
        acc = evaluate(model, 'cpu', test_loader, evaluator)
        results.append({'dataset': dataset_id, 'model': model_id, 'acc': acc})

df = pd.DataFrame(results)
df = df.pivot(index=['model'], columns=['dataset'])
print(df.to_latex())
