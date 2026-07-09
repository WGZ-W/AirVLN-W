import argparse
import os
import pickle
from collections import Counter

import lmdb
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan AirVLN LMDB trajectory data for invalid oracle actions."
    )
    parser.add_argument(
        "--lmdb-path",
        default="/mnt/sdd/weiguanzhao/AirVLN_ws/DATA/img_features/collect/AerialVLN/train",
        help="Path to the LMDB directory.",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=8,
        help="Number of valid action classes. Valid labels are [0, num_actions).",
    )
    parser.add_argument(
        "--max-bad-print",
        type=int,
        default=30,
        help="Maximum number of bad samples and load errors to print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("LMDB_PATH:", args.lmdb_path)
    print("exists:", os.path.exists(args.lmdb_path))
    if not os.path.exists(args.lmdb_path):
        raise FileNotFoundError(args.lmdb_path)

    bad_samples = []
    load_errors = []
    hist = Counter()

    global_min = None
    global_max = None
    total = 0
    empty = 0

    with lmdb.open(
        args.lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=2048,
    ) as env:
        entries = env.stat()["entries"]
        print("entries:", entries)

        with env.begin(buffers=True) as txn:
            cursor = txn.cursor()

            for key_buf, value_buf in tqdm(cursor, total=entries, dynamic_ncols=True):
                total += 1
                key = bytes(key_buf).decode(errors="ignore")

                try:
                    data = pickle.loads(bytes(value_buf))
                    if not isinstance(data, (tuple, list)) or len(data) < 3:
                        load_errors.append(
                            (key, "bad sample structure", type(data).__name__)
                        )
                        continue

                    _obs, _prev_actions, oracle_actions = data[:3]
                    acts = np.asarray(oracle_actions)

                    if acts.size == 0:
                        empty += 1
                        continue

                    sample_min = int(acts.min())
                    sample_max = int(acts.max())

                    global_min = (
                        sample_min if global_min is None else min(global_min, sample_min)
                    )
                    global_max = (
                        sample_max if global_max is None else max(global_max, sample_max)
                    )

                    vals, counts = np.unique(acts, return_counts=True)
                    for val, count in zip(vals.tolist(), counts.tolist()):
                        hist[int(val)] += int(count)

                    bad_mask = (acts < 0) | (acts >= args.num_actions)
                    if bad_mask.any():
                        bad_positions = np.where(bad_mask)[0][:20]
                        bad_samples.append(
                            {
                                "key": key,
                                "shape": tuple(acts.shape),
                                "min": sample_min,
                                "max": sample_max,
                                "first30": acts[:30].tolist(),
                                "bad_positions_first20": bad_positions.tolist(),
                                "bad_values_first20": acts[bad_positions].tolist(),
                            }
                        )

                except Exception as exc:
                    load_errors.append((key, type(exc).__name__, str(exc)))

    print("\n===== SUMMARY =====")
    print("checked entries:", total)
    print("empty oracle_actions:", empty)
    print("global min/max:", global_min, global_max)
    print("action histogram:", dict(sorted(hist.items())))
    print("bad samples:", len(bad_samples))
    print("load errors:", len(load_errors))

    print("\n===== BAD SAMPLES =====")
    for i, item in enumerate(bad_samples[: args.max_bad_print]):
        print(f"BAD[{i}]: {item}")

    if len(bad_samples) > args.max_bad_print:
        print(f"... omitted {len(bad_samples) - args.max_bad_print} bad samples")

    print("\n===== LOAD ERRORS =====")
    for i, item in enumerate(load_errors[: args.max_bad_print]):
        print(f"LOAD_ERROR[{i}]: {item}")

    if len(load_errors) > args.max_bad_print:
        print(f"... omitted {len(load_errors) - args.max_bad_print} load errors")


if __name__ == "__main__":
    main()
