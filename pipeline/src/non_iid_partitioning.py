from typing import List, Tuple
import numpy as np

# Type aliases
XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]

def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle x and y in unison."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]

def sort_by_label(x: np.ndarray, y: np.ndarray) -> XY:
    """Sort (x,y) pairs by label."""
    idx = np.argsort(y, axis=0)
    return x[idx], y[idx]

def _sample_without_replacement(distribution: np.ndarray,
                                class_samples: List[List[np.ndarray]],
                                num_samples: int):
    data, labels = [], []
    for _ in range(num_samples):
        # Find available classes (non-empty)
        available_classes = [i for i in range(len(class_samples)) if len(class_samples[i]) > 0]
        
        if not available_classes:
            raise RuntimeError("No more samples left to draw from any class!")
        
        # If only one class available, choose it directly
        if len(available_classes) == 1:
            chosen_class = available_classes[0]
        else:
            # Sample from the distribution, but only consider available classes
            while True:
                chosen_class = np.argmax(np.random.multinomial(1, distribution))
                if len(class_samples[chosen_class]) > 0:
                    break
                # If chosen class is empty, continue drawing from available classes
                # Update distribution to zero out empty classes
                distribution[chosen_class] = 0.0
                total = distribution.sum()
                if total > 0:
                    distribution /= total

        sample = class_samples[chosen_class].pop()
        data.append(sample)
        labels.append(chosen_class)

        # If we just emptied that class, zero it out and re-normalize if needed
        if len(class_samples[chosen_class]) == 0:
            distribution[chosen_class] = 0.0
            total = distribution.sum()
            if total > 0:
                distribution /= total

    return np.stack(data, axis=0), np.array(labels, dtype=np.int64)

def create_beta_partitions(
    x: np.ndarray,
    y: np.ndarray,
    num_partitions: int,
    alpha: float,
    samples_per_partition: List[int],
    seed: int = 1234
) -> XYList:
    """
    Create `num_partitions` partitions using a Beta distribution for non-IID partitioning.
    
    This function is designed for exactly 2 classes. For each partition, a proportion p is drawn from
    a Beta distribution whose parameters are now determined by the relative class sizes rather than the absolute counts.
    Specifically, we set:
        a_param = alpha * (n0 / total)
        b_param = alpha * (n1 / total)
    where n0 and n1 are the number of samples in class 0 and class 1 respectively, and total = n0 + n1.
    This ensures that the expected proportion of class 0 in the partition is (n0/total), but the variance
    in the drawn p depends solely on alpha.
    
    Partition sizes are given by `samples_per_partition`.
    Leftover samples (if sum(samples_per_partition) < len(x)) are not used.
    When alpha is infinite, an IID partitioning is performed.
    """
    if len(samples_per_partition) != num_partitions:
        raise ValueError("Length of samples_per_partition must match num_partitions.")
    if sum(samples_per_partition) > len(x):
        raise ValueError(
            f"Sum of samples_per_partition ({sum(samples_per_partition)}) exceeds total number of samples ({len(x)})."
        )

    x, y = shuffle(x, y)
    x, y = sort_by_label(x, y)

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(y)
    
    if len(unique_labels) != 2:
        raise ValueError(f"Beta distribution partitioning requires exactly 2 classes, but got {len(unique_labels)}.")

    # Separate samples by class
    _, start_indices = np.unique(y, return_index=True)
    start_indices = np.append(start_indices, len(y))
    
    class_samples: List[List[np.ndarray]] = []
    for c_idx in range(len(unique_labels)):
        start, end = start_indices[c_idx], start_indices[c_idx + 1]
        class_samples.append([x[i] for i in range(start, end)])

    # Shuffle inside each class list for randomness
    for c_idx in range(len(class_samples)):
        rng.shuffle(class_samples[c_idx])

    # If alpha is "infinite", perform an IID split.
    if np.isinf(alpha):
        idx = rng.permutation(len(x))
        x_shuffled, y_shuffled = x[idx], y[idx]
        partitions: XYList = []
        idx_start = 0
        for sz in samples_per_partition:
            idx_end = idx_start + sz
            partitions.append(
                (x_shuffled[idx_start:idx_end], y_shuffled[idx_start:idx_end])
            )
            idx_start = idx_end
        return partitions

    # Determine the number of samples available per class.
    n0 = len(class_samples[0])
    n1 = len(class_samples[1])
    total = n0 + n1
    # Compute relative class proportions.
    prop0 = n0 / total
    prop1 = n1 / total
    # Set Beta distribution parameters based on relative class proportions.
    a_param = alpha * prop0
    b_param = alpha * prop1

    partitions: XYList = []
    for part_idx in range(num_partitions):
        # Draw a proportion p from Beta(a_param, b_param).
        p = rng.beta(a_param, b_param)
        distribution = np.array([p, 1 - p])
        # Sample the desired number of items for this partition.
        part_data, part_labels = _sample_without_replacement(
            distribution, class_samples, samples_per_partition[part_idx]
        )
        partitions.append((part_data, part_labels))
    return partitions