
# ViHSD dataset
```
TF-IDF + SVM (Tuned) — Test
  Accuracy   : 0.9126
  F1 Macro   : 0.7131
  F1 Weighted: 0.9030
  F1   NON-HATE: 0.9523
  F1       HATE: 0.4739

              precision    recall  f1-score   support

    NON-HATE       0.93      0.97      0.95      5992
        HATE       0.62      0.38      0.47       688

    accuracy                           0.91      6680
   macro avg       0.78      0.68      0.71      6680
weighted avg       0.90      0.91      0.90      6680



TF-IDF + LR (Tuned) — Test
  Accuracy   : 0.8910
  F1 Macro   : 0.7393
  F1 Weighted: 0.8972
  F1   NON-HATE: 0.9382
  F1       HATE: 0.5404

              precision    recall  f1-score   support

    NON-HATE       0.96      0.92      0.94      5992
        HATE       0.48      0.62      0.54       688

    accuracy                           0.89      6680
   macro avg       0.72      0.77      0.74      6680
weighted avg       0.91      0.89      0.90      6680

BiLSTM - FastTextVi
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9699    0.8541    0.9083      5992
        HATE     0.3770    0.7689    0.5060       688

    accuracy                         0.8454      6680
   macro avg     0.6735    0.8115    0.7072      6680
weighted avg     0.9088    0.8454    0.8669      6680

PhoBert CNN
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9629    0.9177    0.9398      5992
        HATE     0.4912    0.6919    0.5745       688

    accuracy                         0.8945      6680
   macro avg     0.7271    0.8048    0.7571      6680
weighted avg     0.9143    0.8945    0.9021      6680

PhoBert AmpleHate Origin
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9553    0.9526    0.9540      5992
        HATE     0.5972    0.6119    0.6045       688

    accuracy                         0.9175      6680
   macro avg     0.7762    0.7823    0.7792      6680
weighted avg     0.9184    0.9175    0.9180      6680

PhoBert ViAmpleHate 
              precision    recall  f1-score   support

    NON-HATE     0.9541    0.9574    0.9558      5992
        HATE     0.6177    0.5988    0.6081       688

    accuracy                         0.9205      6680
   macro avg     0.7859    0.7781    0.7819      6680
weighted avg     0.9195    0.9205    0.9200      6680

--------------------------
Summary
Accuracy        : 0.9205   (baseline: 0.9175, Δ=+0.0030)
Macro Precision : 0.7859   (baseline: 0.7762, Δ=+0.0097)
Macro Recall    : 0.7781   (baseline: 0.7823, Δ=-0.0042)
Macro F1        : 0.7819   (baseline: 0.7792, Δ=+0.0027)
F1 (HATE)       : 0.6081   (baseline: 0.6045, Δ=+0.0036)

```

# VOZ-HSD dataset
```
TF-IDF + SVM (Tuned) — Test
  Accuracy   : 0.9641
  F1 Macro   : 0.7831
  F1 Weighted: 0.9609
  F1   NON-HATE: 0.9812
  F1       HATE: 0.5850

              precision    recall  f1-score   support

    NON-HATE       0.97      0.99      0.98      9486
        HATE       0.72      0.49      0.58       514

    accuracy                           0.96     10000
   macro avg       0.85      0.74      0.78     10000
weighted avg       0.96      0.96      0.96     10000

TF-IDF + LR (Tuned) — Test
  Accuracy   : 0.9453
  F1 Macro   : 0.7745
  F1 Weighted: 0.9506
  F1   NON-HATE: 0.9708
  F1       HATE: 0.5783

              precision    recall  f1-score   support

    NON-HATE       0.98      0.96      0.97      9486
        HATE       0.48      0.73      0.58       514

    accuracy                           0.95     10000
   macro avg       0.73      0.84      0.77     10000
weighted avg       0.96      0.95      0.95     10000

BiLSTM - FastTextVi
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9940    0.8626    0.9237     18929
        HATE     0.2721    0.9076    0.4187      1071

    accuracy                         0.8650     20000
   macro avg     0.6330    0.8851    0.6712     20000
weighted avg     0.9553    0.8650    0.8966     20000


PhoBert CNN
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9826    0.9775    0.9801      9486
        HATE     0.6217    0.6809    0.6500       514

    accuracy                         0.9623     10000
   macro avg     0.8021    0.8292    0.8150     10000
weighted avg     0.9641    0.9623    0.9631     10000

PhoBert AmpleHate Origin
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9816    0.9807    0.9812      9486
        HATE     0.6501    0.6615    0.6557       514

    accuracy                         0.9643     10000
   macro avg     0.8159    0.8211    0.8185     10000
weighted avg     0.9646    0.9643    0.9644     10000
```
