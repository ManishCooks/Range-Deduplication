import math
import hashlib
import mmh3
import numpy as np

class SignatureMinHash:
    """
    Computes a 1-bit and n-bit MinHash signature based on sign binarization.
    """
    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        gen = np.random.RandomState(seed)
        
        # We need prime number p for modulo operations
        self.p = (1 << 31) - 1
        self.a = gen.randint(1, self.p, size=num_perm, dtype=np.int64)
        self.b = gen.randint(0, self.p, size=num_perm, dtype=np.int64)
        self._hash_matrix_cache = None

    def compute_signature_1bit(self, vector: np.ndarray) -> bytes:
        active_indices = vector >= np.median(vector)
        if not active_indices.any():
            return np.zeros(self.num_perm // 8, dtype=np.uint8).tobytes()
            
        dim = len(vector)
        if self._hash_matrix_cache is None or self._hash_matrix_cache.shape[0] != dim:
            self._hash_matrix_cache = (np.outer(np.arange(dim), self.a) + self.b) % self.p
            
        min_hashes = self._hash_matrix_cache[active_indices].min(axis=0)
        
        # Binarize and pack bits
        bits = (min_hashes % 2).astype(np.uint8)
        return np.packbits(bits).tobytes()

    def compute_signature_multibit(self, vector: np.ndarray, bit_width: int = 32) -> bytes:
        active_indices = vector >= np.median(vector)
        if not active_indices.any():
            if bit_width == 32:
                return np.zeros(self.num_perm, dtype=np.uint32).tobytes()
            elif bit_width == 64:
                return np.zeros(self.num_perm, dtype=np.uint64).tobytes()
            else:
                return np.zeros(self.num_perm, dtype=np.uint16).tobytes()
            
        dim = len(vector)
        if self._hash_matrix_cache is None or self._hash_matrix_cache.shape[0] != dim:
            self._hash_matrix_cache = (np.outer(np.arange(dim), self.a) + self.b) % self.p
            
        min_hashes = self._hash_matrix_cache[active_indices].min(axis=0)
        
        if bit_width == 32:
            return min_hashes.astype(np.uint32).tobytes()
        elif bit_width == 64:
            return min_hashes.astype(np.uint64).tobytes()
        else:  # 16
            return (min_hashes % (2**16)).astype(np.uint16).tobytes()

    def compute_signature(self, vector: np.ndarray, bit_width: int = 1) -> bytes:
        if bit_width == 1:
            return self.compute_signature_1bit(vector)
        return self.compute_signature_multibit(vector, bit_width)
