# Ministral Alignment Lab: BullshitBench Evaluation

A specialized research project focused on aligning small language models (**Ministral-3B** and **Qwen-4B**) to identify and push back against nonsensical logical premises using **Direct Preference Optimization (DPO)** and **Supervised Fine-Tuning (SFT)** on Apple Silicon (MLX).

## 🚀 Key Results
* **Ministral-3B**: Improved from **4.0%** (Baseline) to **74.2%** (SFT V3) Green Rate.
* **Qwen-4B**: Jumped from **35.0%** (Baseline) to **79.0%** (SFT V1) Green Rate with 100% accuracy in Physics.

## 📦 Project Structure
- `chat_gui.py`: Interactive Streamlit dashboard with real-time reasoning visualization.
- `finetune.py` / `post_eval.py`: Core pipeline for LoRA fine-tuning and evaluation.
- `data/`: Curated dataset of reasoning pairs used for alignment.
- `deepseek_evaluation_report.md`: Full technical audit of model performance.

## 🛠️ Setup
1. **Requirements**: `pip install -r requirements.txt`
2. **Run GUI**: `streamlit run chat_gui.py`

## 📊 Deployment
For sharing within your network, see the [Deployment Guide](deployment_guide.md).

---
*Evaluated using DeepSeek R1 14B as the judge.*
