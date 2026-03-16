---
title: OncoVision-X
emoji: 🫁
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
---

# OncoVision-X: Lung Cancer Detection System

AI-powered lung cancer detection system using Dual-Context Attention Networks (DCA-Net).

## Overview

This application analyzes CT scans to detect and classify lung nodules, providing clinical risk assessments and visualizations.

## How to use

1.  Upload a CT scan file (.mhd, .nii, .nii.gz, .npz, or .npy).
2.  Wait for the analysis to complete.
3.  Review the detected nodules, their malignancy risk, and clinical recommendations.
4.  Interact with the 3D visualization to examine the findings.

## Model Architecture

Developed by the OncoVision team, utilizing:
- **Nodule Analysis Stream**: EfficientNet-B0 with cross-slice attention.
- **Context Analysis Stream**: 3D CNN for surrounding anatomy.
- **Fusion**: Multi-head attention for final prediction.
