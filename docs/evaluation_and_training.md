# Evaluation, Training, and Explainability

Model training for OncoVision-X utilizes a progressive curriculum learning strategy to systematically tackle difficult candidate nodule cases without overwhelming the optimizer or available VRAM resources.

## 9. Training Curriculum & Explainability (XAI) Architecture 

The architecture uniquely implements explainability into its training pipeline, relying on Grad-CAM++ algorithms and explicit Slice Importance derived from the internal attention mechanism.

```mermaid
graph TD
    A[Curriculum Stage 1<br/>Easy Cases] --> B[Curriculum Stage 2<br/>Medium Cases]
    B --> C[Curriculum Stage 3<br/>All Cases]
    
    C --> D[Calculate Model Gradients]
    D --> E[Extract Cross-Slice Attention Weights]
    D --> F[Compute Grad-CAM++ Heatmaps]
    
    E --> G[Slice Importance Scoring]
    F --> H[Visual Regional Emphasis]
    
    G --> I[Radiologist Structured Report]
    H --> I
    
    I --> J[Highlight Abnormalities: Spiculation, Lobulation, etc.]
```

## 10. System Evaluation & Deployment Integrations

The deployment pipeline relies on INT8 optimization to compress the model and accelerate processing. Validating clinical utility involves analyzing Expected Calibration Error (ECE) and True Positives versus False Positives (FPs).

```mermaid
graph TD
    A[PyTorch Subsets Validation] --> B{AUC-ROC > Minimum?}
    B -->|No| C[Refine Hyperparameters & Curriculum]
    C --> A
    
    B -->|Yes| D[Quantization to INT8]
    D --> E[ONNX Export Compilation]
    E --> F[Generate Test Candidate Batch]
    
    F --> G[Run Async Inference]
    G --> H[Check Precision & Recall vs Baseline]
    
    H --> I[Decision Curve Analysis & FPs/Scan Analysis]
    I --> J((Final Production Model Deployment))
```
