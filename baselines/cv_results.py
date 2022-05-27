import pandas as pd
import argparse

# DATASET = 'PROTEINS'
# OUTPATH = '../logs/{}/{}_{}_{}_{}_{}_{}/fold-{}/{}.csv'
OUTPATH = '../logs_classification/{}/{}_{}_{}_{}_{}_{}/fold-{}/{}.csv'


def selection_model(fold_idx, dataset='PROTEINS', gnn_type='gcn'):
    layers_grid = range(1, 4)
    hidden_grid = [64, 128]
    pool_grid = ['sum', 'mean', 'max']
    lr_grid = [0.01, 0.001]
    wd_grid = [0.01, 0.001, 0.0001]
    # dropouts = [0.0, 0.1]

    test_metric = ['test_acc', 'test_loss']
    # metric_list = ['val_acc', 'test_acc']

    param_names = ['layers', 'hidden_dim', 'global_pool', 'lr', 'wd', 'fold_idx']
    all_results = []
    num_results = 0

    for layer in layers_grid:
        for hidden in hidden_grid:
            for global_pool in pool_grid:
                for lr in lr_grid:
                    for wd in wd_grid:
                        path = OUTPATH.format(
                            dataset,
                            gnn_type, layer, hidden, global_pool, lr, wd, fold_idx, 'results')
                        path_log = OUTPATH.format(
                            dataset,
                            gnn_type, layer, hidden, global_pool, lr, wd, fold_idx, 'logs')
                        try:
                            metric = pd.DataFrame(pd.read_csv(path, index_col=0).T)
                            metric = metric.reset_index()
                            metric = metric.drop("index", axis=1)
                            metric_log = pd.read_csv(path_log, index_col=0)
                            # metric = pd.read_csv(path, index_col=0)
                            num_results += 1
                        except Exception:
                            continue
                        # metric = pd.DataFrame(metric_train.iloc[-50:].mean()).T
                        # metric_test_avg = metric_test.iloc[-50:].mean()
                        # for tm in test_metric:
                        #     metric[tm] = metric_test_avg[tm]
                        # print(metric_log)
                        metric['val_acc_std'] = metric_log.iloc[-50:]['val_acc'].std()
                        # metric['test_acc_std'] = metric_test.iloc[-50:]['test_acc'].std()
                        params = [layer, hidden, global_pool, lr, wd, fold_idx]
                        for param, param_name in zip(params, param_names):
                            metric[param_name] = [param]
                            # append to big list
                        # print(metric)
                        all_results.append(metric)
    if num_results == 0:
        return pd.DataFrame([])
    all_results = pd.concat(all_results)
    best_model = all_results.loc[all_results['val_acc'] == all_results['val_acc'].max()]
    best_model = best_model.iloc[[best_model['val_acc_std'].idxmin()]]
    return best_model

def main():
    """
    This functions reads all the experiment results given a parameter grid
    and outputs the scores and best models per fold as well as a global
    score.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--gnn-type', type=str, default='gcn')
    args = parser.parse_args()

    results = []
    for fold_idx in range(1, 11):
        model = selection_model(fold_idx, args.dataset, args.gnn_type)
        results.append(model)

    table = pd.concat(results)
    print(table)
    print("final acc: {}".format(table['test_acc'].mean()))
    print("final acc std: {}".format(table['test_acc'].std()))


if __name__ == "__main__":
    main()
