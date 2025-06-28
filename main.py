import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Pneumonia Detection System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='development',
                             choices=['development', 'production'])
    train_parser.add_argument('--model_type', type=str, default='basic',
                             choices=['basic', 'deeper'])
    train_parser.add_argument('--resume', type=str, default=None)
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model_path', type=str, required=True)
    eval_parser.add_argument('--weights_path', type=str, default=None)
    eval_parser.add_argument('--detailed', action='store_true')
    
    # Prediction command
    pred_parser = subparsers.add_parser('predict', help='Make predictions')
    pred_parser.add_argument('--model_path', type=str, required=True)
    pred_parser.add_argument('--image_path', type=str, default=None)
    pred_parser.add_argument('--image_dir', type=str, default=None)
    pred_parser.add_argument('--gradcam', action='store_true')
    pred_parser.add_argument('--threshold', type=float, default=0.5)
    pred_parser.add_argument('--output_dir', type=str, default='predictions')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute appropriate command
    if args.command == 'train':
        from train import main as train_main
        # Pass arguments to train script
        sys.argv = ['train.py']
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.model_type:
            sys.argv.extend(['--model_type', args.model_type])
        if args.resume:
            sys.argv.extend(['--resume', args.resume])
        train_main()
        
    elif args.command == 'evaluate':
        from evaluate import main as eval_main
        sys.argv = ['evaluate.py', '--model_path', args.model_path]
        if args.weights_path:
            sys.argv.extend(['--weights_path', args.weights_path])
        if args.detailed:
            sys.argv.append('--detailed')
        eval_main()
        
    elif args.command == 'predict':
        from predict import main as pred_main
        sys.argv = ['predict.py', '--model_path', args.model_path]
        if args.image_path:
            sys.argv.extend(['--image_path', args.image_path])
        if args.image_dir:
            sys.argv.extend(['--image_dir', args.image_dir])
        if args.gradcam:
            sys.argv.append('--gradcam')
        if args.threshold != 0.5:
            sys.argv.extend(['--threshold', str(args.threshold)])
        if args.output_dir != 'predictions':
            sys.argv.extend(['--output_dir', args.output_dir])
        pred_main()

if __name__ == '__main__':
    main()