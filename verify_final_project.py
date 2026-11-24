"""Run from any location: python verify_final_project.py /path/to/project
Read-only audit: does not overwrite features, outputs, or notebooks.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv)>1 else Path.cwd().resolve()
if root.name == 'code': root = root.parent
errors=[]
def check(condition, label):
    status = 'PASS' if condition else 'FAIL'
    print(f'[{status}] {label}')
    if not condition: errors.append(label)

def load(rel):
    p=root / rel
    check(p.exists(), f'File exists: {rel}')
    if not p.exists(): raise FileNotFoundError(p)
    return pd.read_csv(p, parse_dates=['date'] if rel.endswith('predictions_2026.csv') or rel.endswith('features.csv') else None)

print('Project:', root)
features=load('data/processed/features.csv')
check(features['date'].is_unique and features['date'].is_monotonic_increasing, 'Feature dates sorted and unique')
check(features['consumption_mwh'].notna().all(), 'No missing evaluation targets')
check(len(features)==3226, f'Frozen feature rows = 3226 (actual {len(features)})')
for period,expected in [('train',2557),('validation',365),('test',304)]:
    count=int((features.period==period).sum())
    check(count==expected,f'{period} rows = {expected} (actual {count})')
check(features['baseline'].notna().all(), 'Baseline available on all modeled days')
print('Baseline fallback rows:', int(features['baseline_uses_fallback'].astype(str).str.lower().isin(['true','1']).sum()))

pred=load('output/tables/final_test_predictions_2026.csv')
pred['date']=pd.to_datetime(pred['date'])
test=features.loc[features.period=='test', ['date','consumption_mwh','baseline']]
check(len(pred)==304 and pred['date'].is_unique, '304 unique final-test prediction dates')
merged=pred.merge(test,on='date',how='outer',indicator=True)
check((merged['_merge']=='both').all(),'Prediction dates match every frozen test target')
check(np.allclose(merged['actual_mwh'],merged['consumption_mwh'],rtol=0,atol=1e-6), 'Test actual values match frozen targets')
error=merged['actual_mwh'].to_numpy()-merged['predicted_mwh'].to_numpy()
check(np.allclose(error,merged['prediction_error_mwh'],rtol=0,atol=1e-6),'Stored signed errors are correct')
mae=float(np.mean(np.abs(error)));rmse=float(np.sqrt(np.mean(error**2)));bias=float(np.mean(error))
baseline_error=merged['actual_mwh'].to_numpy()-merged['baseline'].to_numpy()
base_mae=float(np.mean(np.abs(baseline_error)));base_rmse=float(np.sqrt(np.mean(baseline_error**2)))
print(f'OLS test: n={len(merged)}, MAE={mae:.6f}, RMSE={rmse:.6f}, signed error={bias:.6f} MWh/day')
print(f'Baseline: MAE={base_mae:.6f}, RMSE={base_rmse:.6f} MWh/day')
print(f'MAE improvement: {(1-mae/base_mae)*100:.4f}%')
check(abs(mae-72.587581)<0.02 and abs(rmse-94.630102)<0.02,'Reproduce expected final OLS test metrics')
check(abs(base_mae-161.481377)<0.02,'Reproduce expected baseline test MAE')

selection_path=root/'output/tables/selected_model_config.json'
check(selection_path.exists(),'Model selection config exists')
if selection_path.exists():
    config=json.loads(selection_path.read_text())
    check(config.get('selected_model')=='ols_rolling_3y','2025-selected model is frozen as ols_rolling_3y')

# Run no notebook, no retraining: this independently audits existing outputs.
print('\nSUMMARY:', 'ALL CHECKS PASSED' if not errors else f'{len(errors)} CHECK(S) FAILED')
if errors:
    for e in errors:print(' -',e)
    sys.exit(1)
