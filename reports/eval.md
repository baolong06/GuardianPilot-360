# GuardianPilot Evaluation

Chưa có file CSV nhãn. Tạo `labels,pred` CSV rồi chạy lại:

```bash
python tools/evaluate.py --csv data/preds.csv --out reports/eval.md
```

Targets (PRD 9.2): Face Detection >95%, Eye State >90%, Drowsiness Recall >90%, FP <5%.
