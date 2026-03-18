"""Evaluate a LoRA adapter on the benchmark (skeleton)."""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', help='Path to LoRA adapter')
    args = parser.parse_args()
    print('Would evaluate adapter', args.adapter)

if __name__ == '__main__':
    main()
