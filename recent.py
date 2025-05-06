#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate YOLOv8 model with separate train/val/test/test splits"
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to data YAML file defining train/val/test splits'
    )
    parser.add_argument(
        '--model', type=str, default='yolov8m.pt',
        help='Initial YOLOv8 model path or name (e.g. yolov8n.pt)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch', type=int, default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--imgsz', nargs='+', type=int, default=[1280],
        help='Image size: one integer or two integers (width height)'
    )
    parser.add_argument(
        '--device', type=str, default='0',
        help='Device to use (e.g. "cpu" or GPU index like "0")'
    )
    parser.add_argument(
        '--conf', type=float, default=0.1,
        help='Confidence threshold for validation and test'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from last.pt if available'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(args.data)

    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    train_path = cfg.get('train')
    val_path = cfg.get('val')
    test_path = cfg.get('test')
    if not train_path or not val_path:
        raise ValueError("`train:` and `val:` must be defined in data YAML")

    # Resolve image size
    if len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
    elif len(args.imgsz) == 2:
        imgsz = tuple(args.imgsz)
    else:
        raise ValueError("--imgsz must be one or two integers")

    # Log device information
    if args.device != 'cpu' and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(int(args.device))
    else:
        device_name = 'CPU'
    print(f"Using device: {device_name}\n")

    # Load initial model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print("Model loaded successfully!\n")

    # Training
    project_dir = Path('models')
    run_name = 'recent'
    print("Starting training...")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=imgsz,
        device=args.device,
        save=True,
        save_period=5,
        project=str(project_dir),
        name=run_name,
        resume=args.resume,
        patience=5
    )

    # Load best model
    best_path = project_dir / run_name / 'weights' / 'best.pt'
    print(f"\nLoading best model for evaluation: {best_path}")
    model = YOLO(str(best_path))

    # Validation
    print("\nRunning evaluation on validation set...")
    val_metrics = model.val(
        data=str(data_yaml),
        split='val',
        conf=args.conf,
        save=True,
        save_txt=True,
        project='predictions',
        name='val_eval'
    )
    print("Validation metrics:\n", val_metrics)
    print("Validation outputs saved to ./predictions/val_eval/\n")

    # Test (full evaluation with ground truth + save predictions)
    if test_path:
        print("Running evaluation on test set...")
        test_metrics = model.val(
            data=str(data_yaml),
            split='test',
            conf=args.conf,
            save=True,
            save_txt=True,
            project='predictions',
            name='test_eval'
        )
        print("Test metrics:\n", test_metrics)
        print("Test outputs saved to ./predictions/test_eval/")
    else:
        print("No `test:` path defined in data YAML; skipping test evaluation.")


if __name__ == '__main__':
    main()

