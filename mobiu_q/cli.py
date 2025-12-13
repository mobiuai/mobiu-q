#!/usr/bin/env python3
"""
Mobiu-Q Command Line Interface
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="mobiu-q",
        description="Mobiu-Q: Soft Algebra Optimizer for Quantum Computing"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # activate
    activate_parser = subparsers.add_parser("activate", help="Activate license key")
    activate_parser.add_argument("key", help="Your license key")
    
    # status
    subparsers.add_parser("status", help="Check license status")
    
    # version
    subparsers.add_parser("version", help="Show version")
    
    args = parser.parse_args()
    
    if args.command == "activate":
        from mobiu_q import activate_license
        activate_license(args.key)
    
    elif args.command == "status":
        from mobiu_q import check_status
        check_status()
    
    elif args.command == "version":
        from mobiu_q import __version__
        print(f"Mobiu-Q version {__version__}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
