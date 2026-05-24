import argparse
import json
import os


def load_metrics(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', default=r'.\experiment_summary.md')
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        metrics_path = os.path.join('best_models', run, 'metrics.json')
        rows.append(load_metrics(metrics_path))

    lines = [
        '# Project 1 Experiment Summary',
        '',
        '| Run | Model | Experiment | Best Dev Acc | Test Acc | Test Loss |',
        '| --- | --- | --- | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            '| {run_name} | {model} | {experiment} | {best_dev:.4f} | {test_acc:.4f} | {test_loss:.4f} |'.format(
                run_name=row['run_name'],
                model=row['model'],
                experiment=row['experiment'],
                best_dev=row['best_dev_accuracy'],
                test_acc=row['test_accuracy'],
                test_loss=row['test_loss'],
            )
        )

    if len(rows) >= 2:
        baseline = rows[0]
        lines.extend(['', '## Quick Comparisons', ''])
        for row in rows[1:]:
            delta = row['test_accuracy'] - baseline['test_accuracy']
            lines.append(f"- `{row['run_name']}` vs `{baseline['run_name']}`: test accuracy delta {delta:+.4f}")

    text = '\n'.join(lines) + '\n'
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f'Summary saved to: {args.output}')


if __name__ == '__main__':
    main()
