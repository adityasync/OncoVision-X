#!/usr/bin/env python3
"""
OncoVision-X Demo — Lung Nodule Classification
================================================
Self-contained demo that showcases the model on sample data.
Designed for external presentations and demonstrations.

Usage:
  python demo.py                    # Run full demo with all sample cases
  python demo.py --scan-demo        # Demo with raw CT scan from subset0
  python demo.py --patch-demo       # Demo with pre-extracted patches
  python demo.py --interactive      # Interactive mode: enter your own coordinates
"""

import sys
import os
import time
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner():
    print(f"""
{BOLD}{CYAN}╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║    ██████╗ ███╗   ██╗ ██████╗ ██████╗                              ║
║   ██╔═══██╗████╗  ██║██╔════╝██╔═══██╗                             ║
║   ██║   ██║██╔██╗ ██║██║     ██║   ██║                             ║
║   ██║   ██║██║╚██╗██║██║     ██║   ██║                             ║
║   ╚██████╔╝██║ ╚████║╚██████╗╚██████╔╝                             ║
║    ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝                              ║
║   ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ██╗  ██╗           ║
║   ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ╚██╗██╔╝           ║
║   ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║     ╚███╔╝            ║
║   ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║     ██╔██╗            ║
║    ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ██╔╝ ██╗           ║
║     ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝  ╚═╝           ║
║                                                                    ║
║   Dual-Context Attention Network                                   ║
║   AI-Powered Lung Cancer Detection — LUNA16 Dataset                ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝{RESET}
""")


def print_section(title):
    w = 60
    print(f"\n{BOLD}{BLUE}{'─' * w}")
    print(f"  {title}")
    print(f"{'─' * w}{RESET}")


def print_info(label, value):
    print(f"  {DIM}{label}:{RESET} {value}")


def print_result_card(idx, label, prob, confidence, coords=None, diameter=None, uncertainty=None):
    """Print a visually appealing result card."""
    if label == "MALIGNANT":
        color = RED
        icon = "⚠ "
        bar_char = "█"
    else:
        color = GREEN
        icon = "✓ "
        bar_char = "█"

    # Probability bar
    bar_len = 30
    filled = int(prob * bar_len)
    bar = f"{RED}{bar_char * filled}{DIM}{'░' * (bar_len - filled)}{RESET}"

    print(f"""
  {BOLD}┌─────────────────────────────────────────────┐
  │  {icon}Candidate {idx:>3}                               │
  ├─────────────────────────────────────────────┤{RESET}""")

    if coords is not None:
        print(f"  │  Location:     ({coords[0]:>8.1f}, {coords[1]:>8.1f}, {coords[2]:>8.1f}) mm │")
    if diameter is not None:
        print(f"  │  Diameter:     {diameter:>8.1f} mm                      │")

    print(f"  │  Probability:  {bar}  {BOLD}{color}{prob:.1%}{RESET}  │")
    print(f"  │  Prediction:   {BOLD}{color}{label:>10}{RESET}                      │")
    print(f"  │  Confidence:   {confidence:>10}                      │")

    if uncertainty is not None:
        print(f"  │  Uncertainty:  {uncertainty:>10.4f}                      │")

    print(f"  {BOLD}└─────────────────────────────────────────────┘{RESET}")


def typing_print(text, delay=0.02):
    """Simulate typing effect for dramatic presentation."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# ─────────────────────────────────────────────────────────────
# Known LUNA16 subset0 samples with annotations
# ─────────────────────────────────────────────────────────────
KNOWN_NODULES = [
    {
        "scan": "1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059",
        "coordX": 46.19, "coordY": 48.40, "coordZ": -108.58,
        "diameter_mm": 13.60,
        "description": "Large solid nodule (13.6mm) — high suspicion"
    },
    {
        "scan": "1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492",
        "coordX": -100.57, "coordY": 67.26, "coordZ": -231.82,
        "diameter_mm": 6.44,
        "description": "Medium nodule (6.4mm) — moderate suspicion"
    },
    {
        "scan": "1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059",
        "coordX": 36.39, "coordY": 76.77, "coordZ": -123.32,
        "diameter_mm": 4.34,
        "description": "Small nodule (4.3mm) — lower suspicion"
    },
]


def load_model_for_demo():
    """Load the trained model."""
    import torch
    import yaml

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_section("MODEL INITIALIZATION")
    print_info("Device", f"{device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ''))

    ckpt_path = 'results/checkpoints/best.pth'
    if not Path(ckpt_path).exists():
        print(f"\n  {RED}ERROR: Model checkpoint not found at {ckpt_path}{RESET}")
        print(f"  Make sure 'results/checkpoints/best.pth' exists.")
        sys.exit(1)

    with open('configs/training_config.yaml') as f:
        config = yaml.safe_load(f)

    from src.models.dca_net import DCANet
    model = DCANet(config)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    params = sum(p.numel() for p in model.parameters())

    print_info("Architecture", "OncoVision-X (Dual-Context Attention Network)")
    print_info("Parameters", f"{params:,}")
    print_info("Trained for", f"{epoch} epochs")
    print_info("Checkpoint", ckpt_path)
    print_info("Test AUC-ROC", "0.9555")
    print_info("Test F1-Score", "0.7674")

    return model, device


def predict(model, nodule, context, device):
    """Run single prediction."""
    import torch
    nodule_t = torch.from_numpy(nodule).unsqueeze(0).unsqueeze(0).to(device)
    context_t = torch.from_numpy(context).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(nodule_t, context_t)
        prob = torch.sigmoid(logits.squeeze()).item()

    label = "MALIGNANT" if prob > 0.5 else "BENIGN"
    confidence_val = abs(prob - 0.5) * 2
    if confidence_val > 0.8: conf = "Very High"
    elif confidence_val > 0.6: conf = "High"
    elif confidence_val > 0.4: conf = "Moderate"
    elif confidence_val > 0.2: conf = "Low"
    else: conf = "Very Low"

    return prob, label, conf


# ─────────────────────────────────────────────────────────────
# Demo modes
# ─────────────────────────────────────────────────────────────
def demo_with_patches(model, device):
    """Demo using pre-extracted patches from demo_patches/."""
    import numpy as np

    print_section("PATCH-BASED CLASSIFICATION")
    print(f"  {DIM}Using pre-extracted 3D volumetric patches{RESET}")
    print(f"  {DIM}Input: nodule patch (64³) + context patch (48³){RESET}")

    demo_dir = Path('demo_patches')
    pairs = []

    # Find all patch pairs
    for f in sorted(demo_dir.glob('*_nodule.npz')):
        prefix = f.name.replace('_nodule.npz', '')
        context_f = demo_dir / f"{prefix}_context.npz"
        if context_f.exists():
            pairs.append((f, context_f, prefix))

    if not pairs:
        print(f"\n  {YELLOW}No demo patches found in demo_patches/{RESET}")
        print(f"  Copy patches from the server first.")
        return

    print(f"\n  Found {len(pairs)} sample(s) to classify\n")
    time.sleep(0.5)

    for i, (nod_path, ctx_path, name) in enumerate(pairs):
        label_hint = "Cancer" if "pos" in name else "Non-cancer" if "neg" in name else "Unknown"
        print(f"  {DIM}Loading: {name} (ground truth: {label_hint}){RESET}")

        nodule = np.load(nod_path)['patch'].astype(np.float32)
        context = np.load(ctx_path)['patch'].astype(np.float32)

        prob, label, conf = predict(model, nodule, context, device)
        print_result_card(i + 1, label, prob, conf)
        time.sleep(0.3)


def demo_with_scan(model, device):
    """Demo using a raw CT scan from subset0."""
    import numpy as np
    from predict import load_scan, world_to_voxel, extract_candidate_patches

    print_section("RAW CT SCAN CLASSIFICATION")
    print(f"  {DIM}Processing raw CT scans from LUNA16 dataset{RESET}")
    print(f"  {DIM}Input: .mhd scan file + nodule coordinates (mm){RESET}")

    scan_dir = Path('data/subset0/subset0')
    if not scan_dir.exists():
        print(f"\n  {YELLOW}Subset0 data not found at {scan_dir}{RESET}")
        return

    results = []
    for i, nodule_info in enumerate(KNOWN_NODULES):
        scan_path = scan_dir / f"{nodule_info['scan']}.mhd"
        if not scan_path.exists():
            print(f"  {YELLOW}Scan not found: {scan_path.name}{RESET}")
            continue

        print(f"\n  {DIM}[{i+1}/{len(KNOWN_NODULES)}] {nodule_info['description']}{RESET}")
        typing_print(f"  Loading scan: {scan_path.name[:50]}...", delay=0.01)

        scan_arr, origin, spacing, direction = load_scan(scan_path)
        world_coord = np.array([nodule_info['coordX'], nodule_info['coordY'], nodule_info['coordZ']])
        voxel_coord = world_to_voxel(world_coord, origin, spacing)

        print_info("Scan dimensions", f"{scan_arr.shape}")
        print_info("Voxel spacing", f"({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm")
        print_info("World coords", f"({world_coord[0]:.1f}, {world_coord[1]:.1f}, {world_coord[2]:.1f}) mm")
        print_info("Voxel coords", f"({voxel_coord[0]}, {voxel_coord[1]}, {voxel_coord[2]})")

        nodule, context = extract_candidate_patches(scan_arr, voxel_coord)
        if nodule is None:
            print(f"  {YELLOW}Could not extract patches (boundary issue){RESET}")
            continue

        prob, label, conf = predict(model, nodule, context, device)
        print_result_card(
            i + 1, label, prob, conf,
            coords=world_coord,
            diameter=nodule_info['diameter_mm']
        )
        results.append((label, prob))
        time.sleep(0.5)

    return results


def demo_interactive(model, device):
    """Interactive mode: user enters scan path and coordinates."""
    import numpy as np
    from predict import load_scan, world_to_voxel, extract_candidate_patches

    print_section("INTERACTIVE MODE")
    print(f"  {DIM}Enter a CT scan path and nodule coordinates to classify.{RESET}")
    print(f"  {DIM}Type 'quit' to exit.{RESET}")

    while True:
        print()
        scan_path = input(f"  {CYAN}Scan path (.mhd): {RESET}").strip()
        if scan_path.lower() in ('quit', 'q', 'exit'):
            break

        if not Path(scan_path).exists():
            print(f"  {RED}File not found: {scan_path}{RESET}")
            continue

        coords_str = input(f"  {CYAN}Coordinates (x,y,z in mm): {RESET}").strip()
        try:
            coords = [float(c.strip()) for c in coords_str.split(',')]
            assert len(coords) == 3
        except Exception:
            print(f"  {RED}Invalid coordinates. Use format: x,y,z{RESET}")
            continue

        print(f"\n  Processing...")
        scan_arr, origin, spacing, direction = load_scan(scan_path)
        world_coord = np.array(coords)
        voxel_coord = world_to_voxel(world_coord, origin, spacing)

        nodule, context = extract_candidate_patches(scan_arr, voxel_coord)
        if nodule is None:
            print(f"  {YELLOW}Could not extract patches at these coordinates{RESET}")
            continue

        prob, label, conf = predict(model, nodule, context, device)
        print_result_card(1, label, prob, conf, coords=world_coord)


def print_model_explanation():
    """Print a brief explanation of the model for the audience."""
    print_section("HOW IT WORKS")
    print(f"""
  {BOLD}OncoVision-X Architecture:{RESET}
  The model analyzes lung CT scans using two parallel streams:

  {CYAN}Stream 1 — Nodule Analysis (64³ patch){RESET}
    EfficientNet-B0 processes 2D slices with cross-slice
    attention to capture spatial patterns across the nodule.

  {CYAN}Stream 2 — Context Analysis (48³ patch){RESET}
    A lightweight 3D CNN captures the surrounding anatomy
    to understand the nodule's environment.

  {CYAN}Fusion — Multi-Head Attention{RESET}
    Both streams are fused via attention to make the final
    malignancy prediction (0-100% probability).

  {BOLD}Input formats supported:{RESET}
    • {GREEN}.mhd{RESET} — MetaImage CT scan files (LUNA16, medical standard)
    • {GREEN}.npz{RESET} — Pre-extracted 3D volumetric patches
    • {GREEN}DICOM{RESET} — Can be converted to .mhd with SimpleITK

  {BOLD}Validated performance (LUNA16 test set):{RESET}
    • AUC-ROC:     {BOLD}0.9555{RESET}
    • Sensitivity:  {BOLD}71.7%{RESET} (cancers detected)
    • Specificity:  {BOLD}97.8%{RESET} (false alarm rate: 2.2%)
    • F1-Score:     {BOLD}0.7674{RESET}
""")


def main():
    parser = argparse.ArgumentParser(
        description='OncoVision-X Demo — Lung Nodule Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--scan-demo', action='store_true',
                       help='Demo with raw CT scans from subset0')
    parser.add_argument('--patch-demo', action='store_true',
                       help='Demo with pre-extracted patches')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode: enter your own scans')
    parser.add_argument('--explain', action='store_true',
                       help='Show model architecture explanation')
    parser.add_argument('--no-typing', action='store_true',
                       help='Disable typing animation')

    args = parser.parse_args()

    banner()

    if args.explain:
        print_model_explanation()
        return

    # Load model
    model, device = load_model_for_demo()
    print_model_explanation()

    # Determine which demos to run
    run_all = not args.scan_demo and not args.patch_demo and not args.interactive

    if run_all or args.patch_demo:
        demo_with_patches(model, device)

    if run_all or args.scan_demo:
        demo_with_scan(model, device)

    if args.interactive:
        demo_interactive(model, device)

    # Final summary
    print_section("DEMO COMPLETE")
    print(f"""
  {BOLD}Thank you for watching the OncoVision-X demonstration!{RESET}

  {DIM}For more information:{RESET}
    • Evaluation results: results/evaluation/
    • Model checkpoint:   results/checkpoints/best.pth
    • Documentation:      docs/
    • Full evaluation:    python evaluate.py
""")


if __name__ == '__main__':
    main()
