
# Knowledge Distillation of LLMs for Automatic Scoring of Science Assessments

Welcome to the Knowledge Distillation repository! This project contains a Jupyter notebook for the knowledge distillation of fine-tuned Large Language Models (LLMs) into smaller, more efficient neural networks for automatic scoring of science education assessments.

## Project Overview

### Abstract

This study proposes a method for knowledge distillation (KD) of fine-tuned Large Language Models (LLMs) into smaller, more efficient, and accurate neural networks. We specifically target the challenge of deploying these models on resource-constrained devices. Our methodology involves training the smaller student model (Neural Network) using the prediction probabilities (as soft labels) of the LLM, which serves as a teacher model. This is achieved through a specialized loss function tailored to learn from the LLM’s output probabilities, ensuring that the student model closely mimics the teacher’s performance. To validate the performance of the KD approach, we utilized a large dataset, 7T, containing 6,684 student-written responses to science questions and three mathematical reasoning datasets with student-written responses graded by human experts. We compared accuracy with state-of-the-art (SOTA) distilled models, TinyBERT, and artificial neural network (ANN) models. Results have shown that the KD approach has 3% and 2% higher scoring accuracy than ANN and TinyBERT, respectively, and comparable accuracy to the teacher model. Furthermore, the student model size is 0.03M, 4,000 times smaller in parameters and 10x faster in inferencing than the teacher model and TinyBERT, respectively. The significance of this research lies in its potential to make advanced AI technologies accessible in typical educational settings, particularly for automatic scoring.

### Key Contributions

- **Successful Application of Knowledge Distillation:** Demonstrated the successful application of a novel knowledge distillation strategy, uniquely adapted and optimized for the context of educational content.
- **Model Size and Efficiency:** Achieved a significant reduction in model size and computational requirements without compromising accuracy. The student model, distilled from a fine-tuned BERT teacher model, exhibits a model size that is 4,000 times smaller and demonstrates an inference speed that is ten times faster than that of its teacher counterpart.
- **Performance Validation:** Validated the effectiveness of our KD method against state-of-the-art models like TinyBERT and generic ANN models using a large dataset of 10k student-written responses to science questions.

## Running the Notebooks

To get started with the notebook provided in this repository, follow the instructions below:

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Jupyter Notebook
- PyTorch
- Transformers (HuggingFace library)

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/Knowledge-Distillation.git
   cd Knowledge-Distillation
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

### Notebook Description

1. **Knowledge_Distillation.ipynb:** This notebook details the knowledge distillation process from a fine-tuned LLM to a smaller neural network model. It includes data preprocessing, model training, and evaluation steps.

## Citation

If you use this model or code in your research, please cite our paper:
```
@article{latif2023knowledge,
  title={Knowledge distillation of llm for education},
  author={Latif, Ehsan and Fang, Luyang and Ma, Ping and Zhai, Xiaoming},
  booktitle={International Conference on Artificial Intelligence in Education},
  year={2024}
}
```

Thank you for using our Knowledge Distillation repository! If you have any questions or feedback, please feel free to open an issue in this repository.
