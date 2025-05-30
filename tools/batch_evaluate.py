import os
import subprocess
import argparse
from pathlib import Path
import sys

def run_evaluation(cfg_file, pretrained_model, beam_reduction, output_dir):
    """
    Run evaluation for a specific beam reduction level
    """
    # Create output directory for this beam reduction
    beam_output_dir = os.path.join(output_dir, f'beam_reduction_{beam_reduction}')
    os.makedirs(beam_output_dir, exist_ok=True)
    
    # Construct the evaluation command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        'tools/test.py',
        '--cfg_file', cfg_file,
        '--pretrained_model', pretrained_model,
        '--extra_tag', f'beam_reduction_{beam_reduction}',
        '--eval_tag', f'beam_reduction_{beam_reduction}',
        '--save_to_file'
    ]
    
    # Set environment variable for beam reduction
    env = os.environ.copy()
    env['BEAM_REDUCTION'] = str(beam_reduction)
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print(f"Output for {beam_reduction}% beam reduction:")
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors for {beam_reduction}% beam reduction:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {beam_reduction}% beam reduction:")
        print(f"Exit code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return False
    
    # Try to clear CUDA cache, but don't fail if it doesn't work
    try:
        subprocess.run(['nvidia-smi', '--gpu-reset'], check=False)
    except Exception as e:
        print(f"Warning: Could not reset GPU: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Batch evaluation with beam reduction')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to config file')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--beam_reductions', type=int, nargs='+', default=[0, 25, 50, 75], 
                      help='List of beam reduction percentages')
    
    args = parser.parse_args()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation for each beam reduction level
    success = True
    for beam_reduction in args.beam_reductions:
        print(f"\nEvaluating with {beam_reduction}% beam reduction...")
        if not run_evaluation(args.cfg_file, args.pretrained_model, beam_reduction, args.output_dir):
            success = False
            print(f"Failed to evaluate {beam_reduction}% beam reduction")
            break
        print(f"Completed evaluation for {beam_reduction}% beam reduction\n")
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main() 