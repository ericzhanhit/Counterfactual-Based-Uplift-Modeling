"""Prepare raw-data prefixes for execution checks, not representative estimates.

Reads configs/local/<dataset>.json. X5 purchase history is filtered in chunks to
avoid loading all purchases for a small client smoke check. No imputation/scaling
is performed. Full experiments must use configs/local/<dataset>.json instead.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

def prepare(dataset: str, rows: int = 5000):
    if rows < 100:
        raise ValueError('Use at least 100 rows to retain treatment/outcome groups')
    config = json.loads(Path(f'configs/local/{dataset}.json').read_text())
    destination = Path('data/smoke')/dataset
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'{destination} already contains prepared data; reuse configs/smoke/{dataset}.json')
    destination.mkdir(parents=True, exist_ok=True)
    spec = config['dataset']
    if dataset != 'x5':
        raw = pd.read_csv(spec['path'], nrows=rows)
        target = destination/f'{dataset}.csv'
        raw.to_csv(target, index=False)
        spec['path'] = str(target)
    else:
        train = pd.read_csv(spec['train_path'], nrows=rows)
        ids = set(train.client_id)
        clients = pd.read_csv(spec['clients_path'])
        clients = clients[clients.client_id.isin(ids)]
        purchases_path = destination/'purchases.csv'
        first = True
        columns = ['client_id', 'transaction_id', 'product_id']
        for chunk in pd.read_csv(spec['purchases_path'], usecols=columns, chunksize=250000):
            selected = chunk[chunk.client_id.isin(ids)]
            if not selected.empty:
                selected.to_csv(purchases_path, index=False, mode='w' if first else 'a', header=first)
                first = False
        if first:
            pd.DataFrame(columns=columns).to_csv(purchases_path, index=False)
        for name, df in [('train', train), ('clients', clients)]:
            target = destination/f'{name}.csv'
            df.to_csv(target, index=False)
            spec[f'{name}_path'] = str(target)
        spec['purchases_path'] = str(purchases_path)
    config['experiment'].update(counterfactual_folds=3, output_dir=f'outputs/{dataset}_smoke_run01')
    Path('configs/smoke').mkdir(parents=True, exist_ok=True)
    path = Path(f'configs/smoke/{dataset}.json')
    path.write_text(json.dumps(config, indent=2)+'\n')
    print(f'Prepared execution-check data; run: cf-uplift --config {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['lenta', 'megafon', 'x5'], required=True)
    parser.add_argument('--rows', type=int, default=5000)
    args = parser.parse_args()
    prepare(args.dataset, args.rows)
