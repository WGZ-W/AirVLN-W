


class DDPIWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
    ):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        self.keys = []
        self.seed = 1

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                self.keys.append(key.decode())

        self.length = len(self.keys)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.start = 0
        self.end = self.length

        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = 0 + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)
        logger.warning("END init DDP-Dataset \t rank: {} \t start({}) - end({})".format(self.rank, self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i+1) % 10 == 0:
                        logger.warning("rank: {} \t lmdb load: {} / {}".format(self.rank, i+1, self.preload_size))

                    new_preload.append(
                        # msgpack_numpy.unpackb(
                        #     txn.get(str(self.keys[self.load_ordering.pop()]).encode()),
                        #     raw=False,
                        # )
                        pickle.loads(
                            txn.get(
                                str(
                                    self.keys[self.load_ordering.pop()]
                                ).encode()
                            )
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(self.iter_start, self.iter_end)), self.preload_size)
            )
        )

        return self
