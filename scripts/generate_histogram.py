import os
import argparse
import numpy as np
import ants
from tqdm import tqdm
import pickle


def load_volume(filepath):
    img = ants.image_read(filepath)
    data = img.numpy().astype(np.float32)
    return data


def normalize(data):
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min > 0:
        data = (data - data_min) / (data_max - data_min) * 255.0
    return data


def compute_histogram(data, bins=256, range_val=(0, 256)):
    hist, bin_edges = np.histogram(data.flatten(), bins=bins, range=range_val)
    return hist, bin_edges


def generate_histogram_from_directory(input_dir, bins=256, range_val=(0, 256)):
    nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    if len(nii_files) == 0:
        print(f"No NIfTI files found in {input_dir}")
        return None, None
    
    print(f"Found {len(nii_files)} NIfTI files in {input_dir}")
    
    combined_hist = np.zeros(bins, dtype=np.float32)
    
    for nii_file in tqdm(nii_files, desc='Processing files'):
        input_path = os.path.join(input_dir, nii_file)
        
        try:
            volume = load_volume(input_path)
            volume = normalize(volume)
            
            hist, _ = compute_histogram(volume, bins=bins, range_val=range_val)
            combined_hist += hist
            
        except Exception as e:
            print(f"Error processing {nii_file}: {e}")
            continue
    
    combined_hist = combined_hist / len(nii_files)
    
    return combined_hist, nii_files


def save_histogram(hist, output_path, bin_edges=None):
    histogram_data = {
        'histogram': hist,
        'bin_edges': bin_edges,
        'mean': np.mean(hist),
        'std': np.std(hist)
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(histogram_data, f)
    
    print(f"Saved histogram to: {output_path}")


def load_histogram(histogram_path):
    with open(histogram_path, 'rb') as f:
        histogram_data = pickle.load(f)
    return histogram_data


def main():
    parser = argparse.ArgumentParser(description='Generate histogram from NIfTI files')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory path containing NIfTI files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for histogram file (default: input_dir/histogram.pkl)')
    parser.add_argument('--bins', type=int, default=256,
                        help='Number of histogram bins (default: 256)')
    parser.add_argument('--range', type=int, nargs=2, default=[0, 256],
                        help='Histogram range (default: 0 256)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(args.input, 'histogram.pkl')
    
    hist, files = generate_histogram_from_directory(args.input, bins=args.bins, range_val=tuple(args.range))
    
    if hist is not None:
        bin_edges = np.linspace(args.range[0], args.range[1], args.bins + 1)
        save_histogram(hist, args.output, bin_edges)
        
        print(f"\nHistogram statistics:")
        print(f"  Total files processed: {len(files)}")
        print(f"  Mean histogram value: {np.mean(hist):.4f}")
        print(f"  Std histogram value: {np.std(hist):.4f}")
        print(f"  Max histogram value: {np.max(hist):.4f}")
        print(f"  Min histogram value: {np.min(hist):.4f}")


if __name__ == '__main__':
    main()
