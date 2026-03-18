"""Run a single experiment folder using the config and scripts."""
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir')
    args = parser.parse_args()
    print('Would run experiment in', args.exp_dir)

if __name__ == '__main__':
    main()
