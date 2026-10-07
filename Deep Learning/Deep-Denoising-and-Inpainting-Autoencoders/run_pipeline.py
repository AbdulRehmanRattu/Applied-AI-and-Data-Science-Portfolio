import argparse
import sys
import os
import numpy as np

from autoencoder_restoration.data_loader import load_and_preprocess_dataset
from autoencoder_restoration.trainer import AutoencoderTrainer
from autoencoder_restoration.models.convolutional_autoencoder import get_model_layer_summary_table
from autoencoder_restoration.corruptions import add_gaussian_noise, add_spatial_occlusions, add_combined_corruption


def print_architecture_table(model):
    rows = get_model_layer_summary_table(model)
    print("\n" + "=" * 105)
    print(f"{'Layer / Block Name':<32} {'Filters':<10} {'Kernel':<10} {'Strides':<10} {'Activation':<12} {'BatchNorm':<12} {'Output Dim':<15}")
    print("=" * 105)
    for r in rows:
        print(f"{r['block_type']:<32} {r['filters']:<10} {r['kernel_size']:<10} {r['strides']:<10} {r['activation']:<12} {r['batch_norm']:<12} {r['output_dim']:<15}")
    print("=" * 105 + "\n")


def run_benchmark(train_samples=1500, test_samples=300, epochs=5, batch_size=128, download=False):
    print("\n" + "=" * 80)
    print("EXECUTING MULTI-CORRUPTION AUTOENCODER BENCHMARK SUITE")
    print("=" * 80)
    
    print(f"Loading dataset ({train_samples} train samples, {test_samples} test samples)...")
    x_train, x_test = load_and_preprocess_dataset(subset_train_size=train_samples, 
                                                 subset_test_size=test_samples, 
                                                 download=download)
    
    scenarios = [
        ("Gaussian Denoising (sigma=0.20)", "gaussian", 0.20, (8, 8)),
        ("Spatial Inpainting (8x8 Occlusion)", "occlusion", 0.00, (8, 8)),
        ("Combined Multi-Corruption", "combined", 0.20, (8, 8))
    ]
    
    results = []
    
    for title, mode, noise_fac, occ_sz in scenarios:
        print(f"\nEvaluating: {title}")
        trainer = AutoencoderTrainer(corruption_type=mode, noise_factor=noise_fac, occlusion_size=occ_sz)
        
        if len(results) == 0:
            print_architecture_table(trainer.model)
            
        print(f"Training for {epochs} epochs (batch_size={batch_size})...")
        trainer.fit(x_train, x_val=x_test[:100], epochs=epochs, batch_size=batch_size, verbose=0)
        
        metrics = trainer.evaluate(x_test)
        results.append({
            "task": title,
            "mse": metrics["mse"],
            "psnr_db": metrics["psnr_db"],
            "ssim": metrics["ssim"]
        })
        print(f"  MSE:     {metrics['mse']:.5f}")
        print(f"  PSNR:    {metrics['psnr_db']:.2f} dB")
        print(f"  SSIM:    {metrics['ssim']:.4f}")
        
    print("\n" + "=" * 80)
    print(f"{'Corruption Task':<38} {'MSE (x10^-3)':<15} {'PSNR (dB)':<14} {'SSIM Score':<12}")
    print("=" * 80)
    for r in results:
        print(f"{r['task']:<38} {r['mse']*1000:<15.3f} {r['psnr_db']:<14.2f} {r['ssim']:<12.4f}")
    print("=" * 80 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Deep Convolutional Autoencoder Image Restoration CLI")
    parser.add_argument("--mode", type=str, choices=["gaussian", "occlusion", "combined", "benchmark", "visuals", "all"],
                        default="benchmark", help="Execution mode")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--train-samples", type=int, default=1500, help="Number of training samples")
    parser.add_argument("--test-samples", type=int, default=300, help="Number of testing samples")
    parser.add_argument("--noise-factor", type=float, default=0.20, help="Gaussian noise factor")
    parser.add_argument("--occlusion-size", type=int, nargs=2, default=[8, 8], help="Occlusion patch height width")
    parser.add_argument("--download", action="store_true", help="Download official CIFAR-10 dataset if available")
    
    args = parser.parse_args()
    
    if args.mode == "visuals":
        from generate_visuals import generate_all_visuals
        generate_all_visuals()
    elif args.mode in ["gaussian", "occlusion", "combined"]:
        print(f"\nRunning single-task mode: {args.mode}")
        x_train, x_test = load_and_preprocess_dataset(subset_train_size=args.train_samples,
                                                     subset_test_size=args.test_samples,
                                                     download=args.download)
        trainer = AutoencoderTrainer(corruption_type=args.mode, noise_factor=args.noise_factor,
                                    occlusion_size=tuple(args.occlusion_size))
        print_architecture_table(trainer.model)
        trainer.fit(x_train, x_val=x_test[:100], epochs=args.epochs, batch_size=args.batch_size, verbose=1)
        metrics = trainer.evaluate(x_test)
        print("\nEvaluation Results:")
        print(f"  MSE:     {metrics['mse']:.5f}")
        print(f"  PSNR:    {metrics['psnr_db']:.2f} dB")
        print(f"  SSIM:    {metrics['ssim']:.4f}")
    elif args.mode == "benchmark":
        run_benchmark(train_samples=args.train_samples, test_samples=args.test_samples, 
                      epochs=args.epochs, batch_size=args.batch_size, download=args.download)
    elif args.mode == "all":
        from generate_visuals import generate_all_visuals
        run_benchmark(train_samples=args.train_samples, test_samples=args.test_samples,
                      epochs=args.epochs, batch_size=args.batch_size, download=args.download)
        generate_all_visuals()


if __name__ == "__main__":
    main()
