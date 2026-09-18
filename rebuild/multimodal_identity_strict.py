        self.save(gid, obs)
        self.stats["new"] += 1
        return gid

    def assign(self, obs, sets, commit=True, recovery=False):
        count = len(obs)
        gids = sorted(self.pro)
        if count == 0:
            return []
        floor = float(self.icfg.get("dummy_floor", 0.35))
        cols = len(gids) + count
        mat = np.full((count, cols), floor, np.float32)
        for i, rows in enumerate(sets):
            for j, gid in enumerate(gids):
                item = rows.get(gid)
                if item is not None:
                    mat[i, j] = float(item["score"])
        rr, cc = linear_sum_assignment(-mat)
        take = {}
        for i, j in zip(rr.tolist(), cc.tolist()):
            rows = sets[i]
            ranked = sorted(rows.items(), key=lambda x: x[1]["score"], reverse=True)
            second = float(ranked[1][1]["score"]) if len(ranked) > 1 else 0.0
            item = rows.get(gids[j]) if j < len(gids) else None
            gid = None
            if item is not None:
                selected_gid = int(gids[j])
                pair_second = max(
                    (
                        float(value["score"])
                        for other_gid, value in rows.items()
                        if int(other_gid) != selected_gid
                    ),
                    default=0.0,
                )
                if self.accept(item, pair_second, recovery=recovery):
                    gid = selected_gid
            if gid is None and not recovery and commit:
                best = float(ranked[0][1]["score"]) if ranked else 0.0
                deep = float(ranked[0][1]["deep"]) if ranked else 0.0
                if (
                    best < float(self.icfg.get("new_max", 0.48))
                    and deep < float(self.icfg.get("new_deep_max", 0.54))
                ):
                    gid = self.stage_new(obs[i])
            if gid is not None:
                take[i] = gid
        used = set()
        out = ["PENDING"] * count
        for i in range(count):
            gid = take.get(i)
            if gid is None:
                self.stats["pending"] += 1
                continue
            if gid in used:
                self.stats["duplicate"] += 1
                continue
            used.add(gid)
            out[i] = f"G{gid:06d}"
        for i, gid in take.items():
            if out[i] == "PENDING" or not commit:
                continue
            item = obs[i]
            self.save(gid, item)
            if recovery:
                self.stats["recovery_match"] += 1
            if item.get("face"):
                self.stats["face_match"] += 1
            if len(self.pro[gid]["camera"]) > 1:
                self.stats["cross"] += 1
        return out