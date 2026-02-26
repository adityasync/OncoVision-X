# Architecture: OncoVision-X (DCA-Net)

OncoVision-X utilizes the Dual-Context Attention Network (DCA-Net) to classify lung nodules with high precision and clinical uncertainty quantification. This memory-efficient architecture mimics the workflow of a radiologist by examining both the isolated nodule and the surrounding anatomical structures.

Below are detailed architectural diagrams of the system.

## 1. Overall System Architecture

The core architecture operates via a dual-stream process. Stream 1 extracts nodule-specific features using 2.5D slices, while Stream 2 extracts spatial and anatomical context using a lightweight 3D CNN.

```mermaid
graph TD
    A[Input CT Data] --> B[Candidate Generation & Segmentation]
    B --> C["Nodule Patch<br/>[B, 1, 64, 64, 64]"]
    B --> D["Context Patch<br/>[B, 1, 48, 48, 48]"]

    C --> |Stream 1| E["Nodule Feature Extractor<br/>(2.5D CNN + Cross-Slice Attention)"]
    D --> |Stream 2| F["Anatomical Context Extractor<br/>(Lightweight 3D CNN)"]

    E -->|"[B, 512]"| G[Multi-Head Attention Fusion]
    F -->|"[B, 256]"| G

    G -->|"[B, 256]"| H[Prediction Head]
    H --> I["Malignancy Probability<br/>[B, 1]"]
    
    H --> J[Uncertainty Quantification]
    J --> K[Confidence Score & Flags]
```

## 2. Nodule Stream (2.5D Feature Extractor)

Stream 1 utilizes a 2.5D approach to process the 3D patch as a stack of 2D slices. This significantly reduces GPU memory footprints while capturing critical spatial relationships.

```mermaid
graph LR
    A["Nodule Patch<br/>[B, 1, 64, 64, 64]"] -->|Reshape| B["2D Slices<br/>[B*64, 1, 64, 64]"]
    B --> C[EfficientNet-B0 Backbone]
    C -->|"[B*64, 1280]"| D[Linear Projection]
    D -->|Reshape| E["Feature Volume<br/>[B, 64, 512]"]
    E --> F[Cross-Slice Attention]
    F --> G[1D Temporal Conv block]
    G --> H[Adaptive Avg Pool 1D]
    H --> I["Nodule Features<br/>[B, 512]"]
```

## 3. Anatomical Context Stream

Stream 2 captures lung vasculature, airways, and pleural boundaries using a highly efficient 3D Convolutional Neural Network.

```mermaid
graph TD
    A["Context Patch<br/>[B, 1, 48, 48, 48]"] --> B["3D Conv Block 1<br/>(Stride 2) -> [24^3, 64]"]
    B --> C["3D Conv Block 2<br/>(Stride 2) -> [12^3, 128]"]
    C --> D["3D Conv Block 3<br/>(Stride 2) -> [6^3, 256]"]
    D --> E[Spatial Attention 3D]
    E --> F[Global Average Pooling]
    F --> G[Linear Projection]
    G --> H["Context Features<br/>[B, 256]"]
```

## 4. Multi-Head Attention Fusion

The features from both streams are fused using a Multi-Head Attention mechanism to learn cross-feature dependencies effectively.

```mermaid
graph TD
    A["Nodule Features<br/>[B, 512]"] --> C[Concatenation]
    B["Context Features<br/>[B, 256]"] --> C
    
    C -->|"[B, 768]"| D[Linear Projection to Fused Dim]
    D --> E["Multi-Head Attention<br/>(Self-Attention on Streams)"]
    E --> F[Feed Forward Network]
    F --> G[Layer Normalization]
    G --> H["Fused Vector<br/>[B, 256]"]
```

## 5. Uncertainty Quantification

For clinical trustworthiness, OncoVision-X predicts confidence estimates via Monte Carlo (MC) Dropout.

```mermaid
graph LR
    A[Input Features] --> B{Enable Dropout}
    B -->|Forward 1| C["Prediction 1"]
    B -->|Forward ...| D["Prediction ..."]
    B -->|Forward N| E["Prediction N"]
    
    C --> F[Aggregate Predictions]
    D --> F
    E --> F
    
    F --> G[Calculate Variance]
    F --> H[Calculate Mean Probability]
    
    G --> I[Compute Confidence Score]
    I --> J{Confidence > threshold?}
    J -->|Yes| K[Standard Report]
    J -->|No| L[Flag for Radiologist Review]
```
