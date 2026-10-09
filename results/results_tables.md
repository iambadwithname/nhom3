## Test-set results (macro average)

| Model | Accuracy | Precision | Recall | F1-score | Params | Epochs (best) | Train time (s) |
|---|---|---|---|---|---|---|---|
| Simple CNN | 0.9553 | 0.9557 | 0.9599 | 0.9577 | 0.43M | 18 (10) | 62 |
| Simple CNN + Aug | 0.9537 | 0.9565 | 0.9570 | 0.9566 | 0.43M | 33 (25) | 264 |
| Complex CNN + Aug | 0.9569 | 0.9605 | 0.9594 | 0.9596 | 7.56M | 34 (26) | 370 |
| ResNet-50 TL + Aug | 0.9729 | 0.9744 | 0.9761 | 0.9752 | 23.51M | 28 (20) | 467 |
| MobileNetV3 TL + Aug | 0.9761 | 0.9780 | 0.9785 | 0.9781 | 4.21M | 24 (16) | 278 |

## Per-class test results

| Model | Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|---|
| Simple CNN | COVID | 0.9641 | 1.0000 | 0.9817 | 161 |
| Simple CNN | NORMAL | 0.9437 | 0.9393 | 0.9415 | 214 |
| Simple CNN | PNEUMONIA | 0.9595 | 0.9405 | 0.9499 | 252 |
| Simple CNN + Aug | COVID | 0.9875 | 0.9814 | 0.9844 | 161 |
| Simple CNN + Aug | NORMAL | 0.9148 | 0.9533 | 0.9336 | 214 |
| Simple CNN + Aug | PNEUMONIA | 0.9672 | 0.9365 | 0.9516 | 252 |
| Complex CNN + Aug | COVID | 0.9936 | 0.9689 | 0.9811 | 161 |
| Complex CNN + Aug | NORMAL | 0.9248 | 0.9766 | 0.9500 | 214 |
| Complex CNN + Aug | PNEUMONIA | 0.9631 | 0.9325 | 0.9476 | 252 |
| ResNet-50 TL + Aug | COVID | 0.9938 | 1.0000 | 0.9969 | 161 |
| ResNet-50 TL + Aug | NORMAL | 0.9498 | 0.9720 | 0.9607 | 214 |
| ResNet-50 TL + Aug | PNEUMONIA | 0.9797 | 0.9563 | 0.9679 | 252 |
| MobileNetV3 TL + Aug | COVID | 1.0000 | 0.9938 | 0.9969 | 161 |
| MobileNetV3 TL + Aug | NORMAL | 0.9502 | 0.9813 | 0.9655 | 214 |
| MobileNetV3 TL + Aug | PNEUMONIA | 0.9837 | 0.9603 | 0.9719 | 252 |

## Simple CNN hyperparameter search (validation set)

| Learning rate | Batch size | Val accuracy | Val macro-F1 | Best epoch |
|---|---|---|---|---|
| 0.003 | 32 | 0.9569 | 0.9586 | 25 |
| 0.003 | 64 | 0.9537 | 0.9555 | 20 |
| 0.001 | 32 | 0.9537 | 0.9562 | 20 |
| 0.001 | 64 | 0.9442 | 0.9462 | 23 |
| 0.0003 | 32 | 0.9346 | 0.9375 | 21 |
| 0.0003 | 64 | 0.9250 | 0.9262 | 12 |
