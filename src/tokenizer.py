import collections
from tqdm import tqdm
import numpy as np

class TimeSeriesBPE:
    def __init__(self, vocab_size: int = 1000, initial_vocab_size: int = 26):
        self.vocab_size = vocab_size
        self.initial_vocab_size = initial_vocab_size
        # Vocabulary maps: token_id -> (token_id1, token_id2)
        # Base tokens are not in this map or map to themselves/None essentially.
        self.rules = {} 
        # Reverse map for checking existence
        self.pair_to_token = {}
        # Current valid tokens (starts with 0..initial_vocab_size-1)
        self.current_vocab_count = initial_vocab_size

    def get_stats(self, ids):
        counts = collections.defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def merge_vocab(self, pair, v_in):
        v_out = []
        i = 0
        new_token_id = self.pair_to_token[pair]
        while i < len(v_in):
            if i < len(v_in) - 1 and v_in[i] == pair[0] and v_in[i+1] == pair[1]:
                v_out.append(new_token_id)
                i += 2
            else:
                v_out.append(v_in[i])
                i += 1
        return v_out

    def train(self, data_tokens: list, max_steps: int = None, min_freq: int = 0):
        """
        data_tokens: list of integers (initial discretized tokens)
        min_freq: stop merging if frequency of best pair is less than min_freq (P_min)
        """
        # We work with a list of integers
        ids = list(data_tokens)
        
        target_size = self.vocab_size
        num_merges = target_size - self.initial_vocab_size
        
        if max_steps:
             num_merges = min(num_merges, max_steps)

        pbar = tqdm(range(num_merges), desc="Training BPE")
        for i in pbar:
            stats = self.get_stats(ids)
            if not stats:
                print("No more pairs to merge.")
                break
            
            # Find most frequent pair
            best_pair = max(stats, key=stats.get)
            best_freq = stats[best_pair]

            if best_freq < min_freq:
                print(f"Stopping: Best pair frequency {best_freq} < {min_freq}")
                break
            
            # Assign new token ID
            new_id = self.current_vocab_count
            self.rules[new_id] = best_pair
            self.pair_to_token[best_pair] = new_id
            self.current_vocab_count += 1
            
            # Merge
            ids = self.merge_vocab(best_pair, ids)
            
            pbar.set_postfix({"stats_len": len(stats), "freq": stats[best_pair], "new_id": new_id})

        return ids

    def encode(self, data_tokens: list):
        """
        Apply learned merges to new data
        """
        ids = list(data_tokens)
        # We need to apply merges in the same order they were learned (by ID value)
        # Since ids are assigned sequentially, we can just iterate through our rules sorted by ID.
        sorted_rules = sorted(self.rules.items(), key=lambda x: x[0])
        
        for new_id, pair in sorted_rules:
            ids = self.merge_vocab(pair, ids)
            
        return ids

    def decode_token(self, token_id):
        """
        Recursive decoding of a single token back to base tokens.
        """
        if token_id < self.initial_vocab_size:
            return [token_id]
        
        if token_id not in self.rules:
            # Should not happen in consistent state
            return [token_id]
            
        left, right = self.rules[token_id]
        return self.decode_token(left) + self.decode_token(right)

    def decode(self, tokens: list):
        """
        Decode sequence of BPE tokens back to initial discretization tokens.
        """
        decoded = []
        for t in tokens:
            decoded.extend(self.decode_token(t))
        return decoded
