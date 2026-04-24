import json
from src.evaluation.metrics import evaluate_with_metrics

def evaluate_model(model, test_seen_loader, test_unseen_loader, results_dir, device):

    print("="*60)
    print("Evaluating Model")
    print("="*60)

    # Evaluate on seen noise test set
    print("\nEvaluating on SEEN noise types...")
    results_seen = evaluate_with_metrics(model, test_seen_loader, device=device, desc="Test (Seen)")

    # Evaluate on unseen noise test set
    print("\nEvaluating on UNSEEN noise types...")
    results_unseen = evaluate_with_metrics(model, test_unseen_loader, device=device, desc="Test (Unseen)")

    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)

    print("\nSEEN Noise Types:")
    print(f"  MSE:  {results_seen['mse']:.6f}")
    print(f"  PESQ: {results_seen['pesq_mean']:.3f} ± {results_seen['pesq_std']:.3f}")
    print(f"  STOI: {results_seen['stoi_mean']:.3f} ± {results_seen['stoi_std']:.3f}")

    print("\nUNSEEN Noise Types:")
    print(f"  MSE:  {results_unseen['mse']:.6f}")
    print(f"  PESQ: {results_unseen['pesq_mean']:.3f} ± {results_unseen['pesq_std']:.3f}")
    print(f"  STOI: {results_unseen['stoi_mean']:.3f} ± {results_unseen['stoi_std']:.3f}")

    print("\nGeneralization Gap:")
    print(f"  PESQ: {results_seen['pesq_mean'] - results_unseen['pesq_mean']:.3f}")
    print(f"  STOI: {results_seen['stoi_mean'] - results_unseen['stoi_mean']:.3f}")

    # Save results
    results_summary = {
        'seen': {
            'mse': results_seen['mse'],
            'pesq_mean': results_seen['pesq_mean'],
            'pesq_std': results_seen['pesq_std'],
            'stoi_mean': results_seen['stoi_mean'],
            'stoi_std': results_seen['stoi_std']
        },
        'unseen': {
            'mse': results_unseen['mse'],
            'pesq_mean': results_unseen['pesq_mean'],
            'pesq_std': results_unseen['pesq_std'],
            'stoi_mean': results_unseen['stoi_mean'],
            'stoi_std': results_unseen['stoi_std']
        }
    }

    with open(results_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✓ Results saved to {results_dir / 'evaluation_results.json'}")

    return results_seen, results_unseen