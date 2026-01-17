import torch
from torch.utils.data import Subset

def post_split_diagnostics(
    train_dataset: Subset,
    valid_dataset: Subset,
    group_columns_indices,   # list of ints, optional
):
    def check(condition, name):
        if condition:
            print(f"[PASS] {name}")
            return True
        else:
            print(f"[FAIL] {name}")
            return False

    # Extract features
    train_X = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    valid_X = torch.stack([valid_dataset[i][0] for i in range(len(valid_dataset))])
    
    # Times (last column)    
    train_times = train_X[:, -1]
    valid_times = valid_X[:, -1]

    results = []

    print("\n--- BASIC SANITY ---")
    results.append(check(len(train_times) > 0, "Train set not empty"))
    results.append(check(len(valid_times) > 0, "Valid set not empty"))

    # Split ratio    
    print("\n--- SPLIT RATIO ---")
    total_samples = len(train_times) + len(valid_times)
    valid_ratio = len(valid_times) / total_samples
    print(f"Valid ratio = {valid_ratio:.3f}")
    results.append(check(0.05 <= valid_ratio <= 0.25, "Approx. expected split ratio"))

    # Temporal ordering
    print("\n--- TEMPORAL ORDERING ---")
    results.append(
        check(
            train_times.max() <= valid_times.min(),
            "Train times ≤ Valid times"
        )
    )
    
    # Group integrity
    if group_columns_indices is not None:
        print("\n--- GROUP INTEGRITY ---")
        def extract_groups(X, indices):
            return {tuple(row[indices].tolist()) for row in X}

        train_groups = extract_groups(train_X, group_columns_indices)
        valid_groups = extract_groups(valid_X, group_columns_indices)

        results.append(
            check(
                train_groups.isdisjoint(valid_groups),
                "No group leakage between train and valid"
            )
        )

    # Summary
    print("\n=== SUMMARY ===")
    passed = sum(results)
    total = len(results)
    print(f"Passed {passed}/{total} checks")
    if passed == total:
        print("ALL CHECKS PASSED ✅")
    else:
        print("SOME CHECKS FAILED ⚠️")