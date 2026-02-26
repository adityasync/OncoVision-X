# Data Preprocessing Pipeline

Handling volumetric CT data efficiently is paramount. The OncoVision-X pipeline operates iteratively over the LUNA16 dataset, extracting regions of interest (ROI) and managing severe class imbalance.

## 6. LUNA16 Data Preprocessing Pipeline

The primary system converts massive `.mhd`/`.raw` files into bite-sized compressed `.npz` chunks readable by PyTorch dataloaders.

```mermaid
graph TD
    A["LUNA16 Raw Scans<br/>(.mhd / .raw)"] --> B[Load CT Scan]
    C["annotations.csv<br/>(True Nodules)"] --> B
    D["candidates.csv<br/>(All Candidates)"] --> B
    E["seg-lungs-LUNA16<br/>(Lung Masks)"] --> B
    
    B --> F{Is candidate inside Lung Mask?}
    F -->|No| G[Discard Candidate]
    F -->|Yes| H[Apply HU Windowing]
    
    H --> I["Normalize to [-1, 1]"]
    I --> J["Save to Disk"]
    
    J --> K["preprocessed_data/nodule_patches/"]
    J --> L["preprocessed_data/context_patches/"]
```

## 7. Patch Extraction Strategy

The dual extraction strategy simultaneously samples dense nodule volumes and wider contextual lung anatomy to serve our two distinct architecture streams.

```mermaid
graph LR
    A[Voxel Coordinate Center] --> B[Nodule Window Extraction]
    A --> C[Context Window Extraction]
    
    B -->|64x64x64 resolution| D[Nodule Patch]
    
    C -->|96x96x96 resolution| E[Downsample 2x]
    E -->|48x48x48 resolution| F[Context Patch]
    
    D --> G[Save .npz]
    F --> G
```

## 8. Data Balancing & Split Logic

LUNA16 contains massive class imbalances (approximately 1:1350 positive to negative nodes). The preprocessing framework samples intelligently to achieve a reasonable 1:7 ratio in training.

```mermaid
graph TD
    A[All Processed Candidates] --> B[Positive Samples]
    A --> C[Negative Samples]
    
    C --> D["Rank by distance to nearest positive"]
    D --> E["Hard Negatives<br/>(Select 5x vs Positives)"]
    C --> F["Random Negatives<br/>(Select 2x vs Positives)"]
    
    B --> G["Combined Subsets Pool"]
    E --> G
    F --> G
    
    G -->|Subsets 0, 1, 2| H[Training Set]
    G -->|Subset 3| I[Validation Set]
    G -->|Subset 4| J[Test Set]
```
