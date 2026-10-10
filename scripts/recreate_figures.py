"""Regenerate uplift plots from a completed binary run's actual predictions."""
import argparse
from pathlib import Path
import pandas as pd
from cf_uplift.visualization.uplift_plots import plot_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    root = Path(args.run_directory)
    files = [root/'predictions'/name for name in ['oof.csv', 'test.csv']]
    data = pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
    if data.empty: raise ValueError('No saved predictions')
    plot_predictions(data, root/'figures')

