## Test-set results (macro average)

| Model | Accuracy | Precision | Recall | F1-score | Params | Epochs (best) | Train time (s) |
|---|---|---|---|---|---|---|---|
| Simple CNN | 0.9490 | 0.9492 | 0.9519 | 0.9504 | 15.76M | 12 (4) | 31 |
| Simple CNN + Aug | 0.9474 | 0.9491 | 0.9515 | 0.9498 | 15.76M | 30 (22) | 142 |
| Complex CNN + Aug | 0.9569 | 0.9605 | 0.9594 | 0.9596 | 7.56M | 34 (26) | 370 |
| MobileNetV3 TL + Aug | 0.9761 | 0.9780 | 0.9785 | 0.9781 | 4.21M | 24 (16) | 278 |

## Per-class test results

| Model | Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|---|
| Simple CNN | COVID | 0.9573 | 0.9752 | 0.9662 | 161 |
| Simple CNN | NORMAL | 0.9309 | 0.9439 | 0.9374 | 214 |
| Simple CNN | PNEUMONIA | 0.9593 | 0.9365 | 0.9478 | 252 |
| Simple CNN + Aug | COVID | 0.9691 | 0.9752 | 0.9721 | 161 |
| Simple CNN + Aug | NORMAL | 0.9115 | 0.9626 | 0.9364 | 214 |
| Simple CNN + Aug | PNEUMONIA | 0.9665 | 0.9167 | 0.9409 | 252 |
| Complex CNN + Aug | COVID | 0.9936 | 0.9689 | 0.9811 | 161 |
| Complex CNN + Aug | NORMAL | 0.9248 | 0.9766 | 0.9500 | 214 |
| Complex CNN + Aug | PNEUMONIA | 0.9631 | 0.9325 | 0.9476 | 252 |
| MobileNetV3 TL + Aug | COVID | 1.0000 | 0.9938 | 0.9969 | 161 |
| MobileNetV3 TL + Aug | NORMAL | 0.9502 | 0.9813 | 0.9655 | 214 |
| MobileNetV3 TL + Aug | PNEUMONIA | 0.9837 | 0.9603 | 0.9719 | 252 |

## Simple CNN hyperparameter search (validation set)

| Learning rate | Batch size | Val accuracy | Val macro-F1 | Best epoch |
|---|---|---|---|---|
| 0.003 | 32 | 0.4003 | 0.1906 | 4 |
| 0.003 | 64 | 0.9346 | 0.9374 | 23 |
| 0.001 | 32 | 0.9219 | 0.9236 | 14 |
| 0.001 | 64 | 0.9537 | 0.9558 | 22 |
| 0.0003 | 32 | 0.9394 | 0.9418 | 14 |
| 0.0003 | 64 | 0.9426 | 0.9451 | 14 |
