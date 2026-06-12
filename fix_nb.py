import json
with open('notebooks/full_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'xgb_calibrated = calibrate_model(xgb_model' in line:
                source.insert(i, 'xgb_model.set_params(early_stopping_rounds=None)\n')
                break

with open('notebooks/full_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
