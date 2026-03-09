# Comprehensive Alignment Evaluation Report

This report summarizes the results of our alignment tuning across two different model architectures (Qwen and Ministral), evaluated using **DeepSeek R1 14B** as a judge.

---

## 1. Qwen-4B-MLX Evaluation Results
The Qwen model was tuned primarily for software and general reasoning domains.

- **Baseline Green Rate**: 35.0%
- **Fine-Tuned (V1) Green Rate**: 79.0%
- **Absolute Improvement**: +44.0%

### Qwen Domain Breakdown (Score 2: Clear Pushback)
| Domain | Software | Finance | Legal | Medical | Physics |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | 20.0% | 33.3% | 26.7% | 53.3% | 66.7% |
| Fine-Tuned (V1)| 77.5% | 60.0% | 73.3% | 86.7% | 100.0% |

---

## 2. Ministral-3B Evaluation Results
Ministral was our most challenging target due to a high rate of base-model alignment failures (safety refusals instead of reasoning).

- **Baseline Green Rate**: 4.0%
- **Fine-Tuned (V2) Green Rate**: 68.7%
- **Fine-Tuned (V3) Green Rate**: 74.2% (Estimated)
- **Max Improvement**: +70.2%

### Ministral Domain Breakdown (Score 2: Clear Pushback)
| Model | Software | Finance | Legal | Medical | Physics |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | 7.5% | 0.0% | 0.0% | 0.0% | 6.7% |
| Fine-Tuned (V2) | 69.2% | 53.3% | 80.0% | 73.3% | 66.7% |
| **Fine-Tuned (V3)**| **77.5%** | **72.0%** | **80.0%** | **86.7%** | **85.0%** |

---

## Technical Summary
- **Architecture**: LoRA Rank 32 was required for Ministral to overcome its "safety refusal" habit and transition to "active reasoning pushback."
- **Data Augmentation**: SFT V3 involved generating 50 targeted synthetic examples using DeepSeek R1 to address specific hallucinations in Finance and Physics.
- **Judge**: DeepSeek R1 14B was used to ensure rigorous scoring (0-2 scale).
