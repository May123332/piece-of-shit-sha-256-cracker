import hashlib
import sys
import argparse
import string
import itertools
import time

# Supported algorithms in hashlib (common ones)
ALGORITHMS = {
    'md5': hashlib.md5,
    'sha1': hashlib.sha1,
    'sha224': hashlib.sha224,
    'sha256': hashlib.sha256,
    'sha384': hashlib.sha384,
    'sha512': hashlib.sha512,
}

def identify_hash(hash_str):
    """
    Attempts to identify the hash type based on length.
    Returns a list of potential algorithms.
    """
    length = len(hash_str)
    possibilities = []

    if length == 32:
        possibilities.append('md5')
    if length == 40:
        possibilities.append('sha1')
    if length == 56:
        possibilities.append('sha224')
    if length == 64:
        possibilities.append('sha256')
    if length == 96:
        possibilities.append('sha384')
    if length == 128:
        possibilities.append('sha512')
    
    return possibilities

def crack_dictionary(hash_str, algo_name, wordlist_path):
    """
    Attempts to crack the hash using a wordlist.
    """
    print(f"[*] Starting Dictionary Attack using {algo_name}...")
    hash_func = ALGORITHMS[algo_name]
    
    try:
        with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word = line.strip()
                # Hash the word
                digest = hash_func(word.encode()).hexdigest()
                if digest == hash_str.lower():
                    return word
    except FileNotFoundError:
        print(f"[!] Wordlist file not found: {wordlist_path}")
        return None
    except Exception as e:
        print(f"[!] Error reading wordlist: {e}")
        return None
    
    return None

def crack_bruteforce(hash_str, algo_name, max_length, charset):
    """
    Attempts to crack the hash using brute force.
    """
    print(f"[*] Starting Brute Force Attack using {algo_name}...")
    print(f"[*] Charset: {charset}")
    print(f"[*] Max Length: {max_length}")
    
    hash_func = ALGORITHMS[algo_name]
    
    start_time = time.time()
    
    for length in range(1, max_length + 1):
        print(f"[*] Checking length {length}...")
        for attempt in itertools.product(charset, repeat=length):
            word = "".join(attempt)
            digest = hash_func(word.encode()).hexdigest()
            
            if digest == hash_str.lower():
                return word
    
    return None

def main():
    parser = argparse.ArgumentParser(description="ShadowBreaker: Local Hash Cracker & Identifier")
    parser.add_argument("hash", help="The hash string to crack")
    parser.add_argument("-w", "--wordlist", help="Path to wordlist file (for dictionary attack)")
    parser.add_argument("-b", "--bruteforce", action="store_true", help="Enable brute force mode if dictionary fails (or if no wordlist provided)")
    parser.add_argument("-l", "--length", type=int, default=4, help="Max length for brute force (default: 4). WARNING: Increasing this increases time exponentially.")
    parser.add_argument("--chars", default=string.ascii_letters + string.digits + string.punctuation, help="Custom charset for brute force")
    
    args = parser.parse_args()
    
    target_hash = args.hash.strip()
    
    # 1. Identify Hash
    print(f"[*] Analyzing hash: {target_hash}")
    potential_algos = identify_hash(target_hash)
    
    if not potential_algos:
        print("[!] Could not identify hash type based on length. It might be a non-standard hash or salted.")
        sys.exit(1)
        
    print(f"[*] Detected potential algorithms: {', '.join(potential_algos)}")
    
    found_password = None
    
    # Iterate through potential algorithms (usually just one, but MD5/others can overlap if custom)
    for algo in potential_algos:
        print(f"\n--- Trying Algorithm: {algo} ---")
        
        # 2. Dictionary Attack
        if args.wordlist:
            found_password = crack_dictionary(target_hash, algo, args.wordlist)
            if found_password:
                print(f"\n[SUCCESS] Password found: {found_password}")
                print(f"[INFO] Algorithm: {algo}")
                sys.exit(0)
        
        # 3. Brute Force Attack
        if args.bruteforce or not args.wordlist:
            if not args.wordlist:
                print("[!] No wordlist provided, defaulting to Brute Force.")
            
            # Warn user if brute force might take long
            if args.length > 5:
                 print(f"[!] WARNING: Brute forcing length {args.length} with full symbols will take a VERY long time.")
            
            found_password = crack_bruteforce(target_hash, algo, args.length, args.chars)
            if found_password:
                print(f"\n[SUCCESS] Password found: {found_password}")
                print(f"[INFO] Algorithm: {algo}")
                sys.exit(0)

    print("\n[FAILED] Could not crack the hash with the provided methods.")

if __name__ == "__main__":
    main()
