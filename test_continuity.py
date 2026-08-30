import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.continuity import MANIFEST_NAME, validate_manifest, write_manifest


class ContinuityManifestTests(unittest.TestCase):
    def test_round_trip_records_and_validates_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "neural" / "latest.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"model")
            manifest = write_manifest(root, seed=2026, run_id="42", source_sha="abc")
            self.assertEqual(manifest.name, MANIFEST_NAME)
            payload = validate_manifest(root)
            self.assertEqual(payload["seed"], 2026)
            self.assertEqual(payload["files"][0]["path"], "neural/latest.pt")

    def test_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "tactical" / "checkpoint.json"
            checkpoint.parent.mkdir()
            checkpoint.write_text('{"generation": 1}')
            write_manifest(root, seed=7, run_id="1", source_sha="abc")
            checkpoint.write_text('{"generation": 2}')
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                validate_manifest(root)

    def test_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / MANIFEST_NAME).write_text(json.dumps({
                "schema_version": 1,
                "seed": 1,
                "run_id": "1",
                "files": [{"path": "../escape.pt", "sha256": hashlib.sha256(b"x").hexdigest()}],
            }))
            with self.assertRaisesRegex(ValueError, "unsafe checkpoint path"):
                validate_manifest(root)


if __name__ == "__main__":
    unittest.main()
