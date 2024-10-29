import argparse
import torch
from detection3d.core.lmk_det_infer import detection

def main():
    parser = argparse.ArgumentParser(description="Run landmark detection")
    parser.add_argument("--input", required=True, help="Input folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--model", required=True, help="Model folder")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--return_landmark_file", type=bool, default=False, help="Whether to return landmark file")
    parser.add_argument("--save_landmark_file", type=bool, default=True, help="Whether to save landmark file")
    parser.add_argument("--save_prob", action="store_true", help="Whether to save probability maps")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Set up CUDA
    torch.cuda.set_device(args.gpu)

    if args.debug:
        print("Debug mode enabled")
        print(f"Input folder: {args.input}")
        print(f"Output folder: {args.output}")
        print(f"Model folder: {args.model}")
        print(f"GPU ID: {args.gpu}")
        print(f"Return landmark file: {args.return_landmark_file}")
        print(f"Save landmark file: {args.save_landmark_file}")
        print(f"Save probability maps: {args.save_prob}")

    # Run inference
    detection(args.input, args.model, args.gpu, args.return_landmark_file, 
              args.save_landmark_file, args.save_prob, args.output)

if __name__ == "__main__":
    main()