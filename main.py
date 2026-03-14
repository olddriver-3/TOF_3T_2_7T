import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='AU-MIPGAN: 7T-like TOF-MRA Generation from 3T')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    train_teacher_parser = subparsers.add_parser('train_teacher', help='Train teacher models')
    train_teacher_parser.add_argument('--direction', type=str, default='axial',
                                       choices=['axial', 'coronal', 'sagittal'],
                                       help='MIP direction for training')
    train_teacher_parser.add_argument('--config', type=str, default=None,
                                       help='Path to config file')
    train_teacher_parser.add_argument('--gpu', type=str, default='0',
                                       help='GPU ID to use (e.g., 0)')
    
    train_student_parser = subparsers.add_parser('train_student', help='Train student model')
    train_student_parser.add_argument('--teacher_axial', type=str, default=None,
                                       help='Path to axial teacher checkpoint')
    train_student_parser.add_argument('--teacher_coronal', type=str, default=None,
                                       help='Path to coronal teacher checkpoint')
    train_student_parser.add_argument('--teacher_sagittal', type=str, default=None,
                                       help='Path to sagittal teacher checkpoint')
    train_student_parser.add_argument('--config', type=str, default=None,
                                       help='Path to config file')
    train_student_parser.add_argument('--gpu', type=str, default='0',
                                      help='GPU ID to use (e.g., 0)')
    
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--input', type=str, required=True,
                                   help='Input file or directory path')
    inference_parser.add_argument('--output', type=str, default=None,
                                   help='Output file or directory path')
    inference_parser.add_argument('--checkpoint', type=str, required=True,
                                   help='Path to model checkpoint')
    inference_parser.add_argument('--config', type=str, default=None,
                                   help='Path to config file')
    inference_parser.add_argument('--mode', type=str, default='file', choices=['file', 'dir'],
                                   help='Inference mode: file or directory')
    inference_parser.add_argument('--gpu', type=str, default='0',
                                  help='GPU ID to use (e.g., 0)')
    
    test_parser = subparsers.add_parser('test', help='Run complete test pipeline with vessel segmentation and metrics')
    test_parser.add_argument('--input', type=str, required=True,
                             help='Input 3T file path or directory containing 3T/7T subdirectories')
    test_parser.add_argument('--gt_7t', type=str, default=None,
                             help='Ground truth 7T file path (required for single file mode)')
    test_parser.add_argument('--output', type=str, required=True,
                             help='Output directory')
    test_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')
    test_parser.add_argument('--config', type=str, default=None,
                             help='Path to config file')
    test_parser.add_argument('--mode', type=str, default='file', choices=['file', 'batch'],
                             help='Test mode: file (single file) or batch (directory)')
    test_parser.add_argument('--gpu', type=str, default='0',
                             help='GPU ID to use')
    test_parser.add_argument('--no_save_intermediate', action='store_true',
                             help='Do not save intermediate results')
    
    args = parser.parse_args()
    
    if args.command == 'train_teacher':
        from scripts.train_teacher import main as train_teacher_main
        sys.argv = ['train_teacher.py', '--direction', args.direction, '--gpu', args.gpu]
        if args.config:
            sys.argv.extend(['--config', args.config])
        train_teacher_main()
    
    elif args.command == 'train_student':
        from scripts.train_student import main as train_student_main
        sys.argv = ['train_student.py', '--gpu', args.gpu]
        if args.teacher_axial:
            sys.argv.extend(['--teacher_axial', args.teacher_axial])
        if args.teacher_coronal:
            sys.argv.extend(['--teacher_coronal', args.teacher_coronal])
        if args.teacher_sagittal:
            sys.argv.extend(['--teacher_sagittal', args.teacher_sagittal])
        if args.config:
            sys.argv.extend(['--config', args.config])
        train_student_main()
    
    elif args.command == 'inference':
        from scripts.inference import main as inference_main
        sys.argv = ['inference.py', '--input', args.input, '--checkpoint', args.checkpoint,
                    '--mode', args.mode, '--gpu', args.gpu]
        if args.output:
            sys.argv.extend(['--output', args.output])
        if args.config:
            sys.argv.extend(['--config', args.config])
        inference_main()
    
    elif args.command == 'test':
        from scripts.test_pipeline import main as test_main
        sys.argv = ['test_pipeline.py', '--input', args.input, '--output', args.output,
                    '--checkpoint', args.checkpoint, '--mode', args.mode, '--gpu', args.gpu]
        if args.gt_7t:
            sys.argv.extend(['--gt_7t', args.gt_7t])
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.no_save_intermediate:
            sys.argv.append('--no_save_intermediate')
        test_main()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
