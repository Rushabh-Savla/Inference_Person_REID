            raise RuntimeError(f"Cannot open video: {path}")

        recovery_frames = max(1, int(self.cfg["identity"].get("recovery_frames", 10)))
        labels = []
        previous_overlap = set()
        recovery_until = -1
        frame = 0

        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                frame += 1
                current = byframe.get(frame, [])
                if not current:
                    continue

                tids = [int(item["track_id"]) for item in current]
                if len(tids) != len(set(tids)):
                    raise RuntimeError(
                        f"NvDCF emitted duplicate tracker IDs in one frame: {camera}:{frame}:{tids}"
                    )

                active_overlap = self._overlap_ids(current)
                if previous_overlap - active_overlap:
                    recovery_until = max(recovery_until, frame + recovery_frames)
                recovery_mode = frame <= recovery_until

                # Every overlap frame still runs the complete feature stack, but
                # its assignments are quarantined. Nothing learned from a
                # mixed/occluded crop is committed to the identity gallery.
                #
                # After overlap, recovery=True forces feature-only identity
                # assignment. NvDCF track_id is never used to choose the GID.
                feature_map = self.identity.observe(
                    image,
                    current,
                    commit=not bool(active_overlap),
                    recovery=bool(active_overlap) or recovery_mode,
                )
                if active_overlap:
                    gids = {
                        int(item["track_id"]): "PENDING"
                        for item in current
                    }
                else:
                    gids = {
                        int(item["track_id"]): str(
                            feature_map.get(int(item["track_id"]), "PENDING")
                        )
                        for item in current
                    }

                # Hard same-frame collision invariant. Re-solve the whole frame
                # with feature-only recovery, then keep any unresolved collision
                # as PENDING instead of emitting a false merge.
                grouped = {}
                for tid, gid in gids.items():
                    if gid.startswith("G"):
                        grouped.setdefault(gid, []).append(tid)
                if any(len(items) > 1 for items in grouped.values()):
                    self.identity.stats["duplicate_frames"] += 1
                    feature_map = self.identity.observe(image, current, commit=False, recovery=True)
                    gids = {int(item["track_id"]): str(feature_map.get(int(item["track_id"]), "PENDING")) for item in current}

                used = set()
                for tid in list(gids):
                    gid = gids[tid]
                    if gid.startswith("G") and gid in used:
                        gids[tid] = "PENDING"
                    elif gid.startswith("G"):
                        used.add(gid)

                for item in current:
                    tid = int(item["track_id"])
                    labels.append({
                        "camera": camera,
                        "frame": frame,
                        "track_id": tid,
                        "bbox": item["bbox"],
                        "gid": gids[tid],
                        "overlap": tid in active_overlap,
                        "recovery": bool(recovery_mode),
                    })

                previous_overlap = set(active_overlap)
        finally:
            cap.release()

        by_frame = {}
        for item in labels:
            by_frame.setdefault(int(item["frame"]), []).append(item)
        for frame_id, items in by_frame.items():
            gids = [str(item["gid"]) for item in items if str(item["gid"]).startswith("G")]
            if len(gids) != len(set(gids)):
                raise RuntimeError(f"same-frame duplicate GID survived validation: {camera}:{frame_id}")

        target = self.cache / f"{camera}.labels.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for item in labels:
                handle.write(json.dumps(item) + "\n")
        output = self._render(camera, path, labels)
        print(f"[nvdcf] wrote {output}")
        return labels

    def run(self, values):
        sources = self.sources(values)
        if not sources:
            raise SystemExit("No videos supplied")

        print("[nvdcf] PRIMARY TRACKER: NVIDIA NvDCF")
        print(f"[nvdcf] ResNet: {self.identity.resnet.describe()}")
        print(f"[nvdcf] Swin: {self.identity.swin.describe()}")
        print(f"[nvdcf] SOLIDER: {self.identity.solider.describe()}")
        print(f"[nvdcf] FACE: {self.identity.face.describe()}")
        print(f"[nvdcf] POSE: {self.cfg['pose']['model']}")
        print("[nvdcf] QDRANT: ENABLED")
        print("[nvdcf] GID assignment: feature-only after overlap and one-to-one every frame")

        all_labels = []
        try:
            for camera, path in sources:
                target = self.cache / f"{camera}.tracker.jsonl"
                fps, width, height = self.det.track(camera, path, target)
                del fps, width, height
                all_labels.extend(self._solve_camera(camera, path, self._load(target)))

            debug = {
                "tracker": "NVIDIA NvDCF",
                "identity": "Qdrant + reliable face + top clothing + bottom clothing + NVIDIA ResNet + NVIDIA Swin + SOLIDER + pose",
                "face_visibility_threshold": float(self.identity.face_threshold),
                "post_overlap_identity": "feature_only",
                "tracker_id_global_fallback": False,
                "same_frame_gid_invariant": True,
                "new_gids": int(self.identity.stats["new"]),
                "duplicate_frames": int(self.identity.stats["duplicate"]),
                "pending_frames": int(self.identity.stats["pending"]),
                "recovery_frames": int(self.identity.stats["recovery"]),
                "recovery_matches": int(self.identity.stats["recovery_match"]),
                "cross_camera_matches": int(self.identity.stats["cross"]),
                "pending_new_observations": int(self.identity.stats["pending_new_observations"]),
                "pending_new_confirmed": int(self.identity.stats["pending_new_confirmed"]),
            }
            (self.out / "nvdcf_identity_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
            print(f"[nvdcf] result: new_gids={debug['new_gids']} recovery_matches={debug['recovery_matches']} cross_camera_matches={debug['cross_camera_matches']} duplicate_frames={debug['duplicate_frames']} pending_frames={debug['pending_frames']}")
            return all_labels
        finally:
            self.identity.close()


__all__ = ["BatchNvDCF"]