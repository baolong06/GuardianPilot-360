import json
import sys

nb = json.load(open('results/notebook6672d603fa.ipynb', 'r', encoding='utf-8'))
sys.stdout.reconfigure(encoding='utf-8')

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if any(k in src for k in ['HYSTERESIS', 'EMA_ALPHA', 'MIN_ON', 'MIN_OFF', 'neck_alarm', 'EMA_PROB_ON', 'neck_baseline', 'neck_recovered', 'def fusion', 'class.*Fusion', 'def __init__']):
        print(f'\n=== CELL {i} ({c["cell_type"]}) ===')
        print(src[:2500])
        print('---')
