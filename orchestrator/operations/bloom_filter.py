import math
import hashlib
import numpy as np

class SignatureMinHash:
    """
    Computes a 1-bit MinHash signature based on sign binarization.
    """
    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        gen = np.random.RandomState(seed)
        
        # We need prime number p for modulo operations
        self.p = (1 << 31) - 1
        self.a = gen.randint(1, self.p, size=num_perm, dtype=np.int64)
        self.b = gen.randint(0, self.p, size=num_perm, dtype=np.int64)
        self._hash_matrix_cache = None

    def compute_signature(self, vector: np.ndarray) -> bytes:
        """
        1. Sign binarize the embedding -> active indices
        2. Compute MinHash -> array of u32
        3. Convert to 1-bit minhash -> bit array
        """
        active_indices = vector > 0
        if not active_indices.any():
            return np.zeros(self.num_perm // 8, dtype=np.uint8).tobytes()
            
        dim = len(vector)
        if self._hash_matrix_cache is None or self._hash_matrix_cache.shape[0] != dim:
            self._hash_matrix_cache = (np.outer(np.arange(dim), self.a) + self.b) % self.p
            
        min_hashes = self._hash_matrix_cache[active_indices].min(axis=0)
        
        # Binarize and pack bits
        bits = (min_hashes % 2).astype(np.uint8)
        return np.packbits(bits).tobytes()

class BloomFilter:
    """
    Standard in-memory Bloom filter.
    """
    def __init__(self, capacity: int, error_rate: float):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_size = self._get_size(capacity, error_rate)
        self.hash_count = self._get_hash_count(self.bit_size, capacity)
        self.bit_array = bytearray((self.bit_size + 7) // 8)
        self.inserted_elements = 0
        
    def _get_size(self, n, p):
        return int(-(n * math.log(p)) / (math.log(2)**2))
        
    def _get_hash_count(self, m, n):
        return max(1, int((m / n) * math.log(2)))
        
    def add(self, item_bytes: bytes):
        for i in range(self.hash_count):
            digest = hashlib.sha256(item_bytes + i.to_bytes(4, 'little')).digest()
            idx = int.from_bytes(digest, 'little') % self.bit_size
            self.bit_array[idx // 8] |= (1 << (idx % 8))
        self.inserted_elements += 1
            
    def __contains__(self, item_bytes: bytes) -> bool:
        for i in range(self.hash_count):
            digest = hashlib.sha256(item_bytes + i.to_bytes(4, 'little')).digest()
            idx = int.from_bytes(digest, 'little') % self.bit_size
            if (self.bit_array[idx // 8] & (1 << (idx % 8))) == 0:
                return False
        return True
