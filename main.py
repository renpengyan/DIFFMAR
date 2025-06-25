import argparse

def main():
    parser = argparse.ArgumentParser(description='DiffMAR: Metal Artifact Reduction with Diffusion Models')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'],
                        help='train or infer')
    args = parser.parse_args()

    if args.mode == 'train':
        from train import train
        train()

    elif args.mode == 'infer':
        from infer import inference
        inference()


if __name__ == '__main__':
    main()
