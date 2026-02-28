#!/usr/bin/env python3
"""
Download the LUNA16 dataset using kagglehub.
This dataset is used for lung cancer detection research.
"""

import kagglehub

# ─────────────────────────────────────────────────────────────
# Display
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
║   AI-Powered Lung Cancer Detection — Dataset Download              ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝{RESET}
""")


def section(title):
    print(f"\n{BOLD}{BLUE}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{RESET}")


def info(label, value):
    print(f"  {DIM}{label}:{RESET} {value}")


def success(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")


def warn(msg):
    print(f"  {YELLOW}! {msg}{RESET}")


def main():
    banner()
    section("KAGGLE DOWNLOAD")
    info("Dataset", "avc0706/luna16")
    warn("This dataset is ~60GB and may take a while to download.")
    
    # Download latest version of the LUNA16 dataset
    path = kagglehub.dataset_download("avc0706/luna16")
    
    section("DOWNLOAD COMPLETE")
    success("Files successfully retrieved!")
    info("Path to files", str(path))
    
    return path

if __name__ == "__main__":
    main()
