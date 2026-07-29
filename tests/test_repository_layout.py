import csv
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTest(unittest.TestCase):
    def test_paper_implementations_exist(self):
        required = (
            REPO_ROOT
            / "third_party"
            / "PUT"
            / "image_synthesis"
            / "modeling"
            / "models"
            / "masked_image_inpainting_transformer.py",
            REPO_ROOT
            / "third_party"
            / "baselines"
            / "latent-code-inpainting"
            / "core"
            / "modules"
            / "transformer"
            / "mingpt.py",
            REPO_ROOT / "tools" / "evaluate_put.py",
            REPO_ROOT / "tools" / "evaluate_latent_codes.py",
        )
        for path in required:
            self.assertTrue(path.is_file(), path)

    def test_paper_settings_are_present(self):
        put_source = (
            REPO_ROOT
            / "third_party"
            / "PUT"
            / "image_synthesis"
            / "modeling"
            / "models"
            / "masked_image_inpainting_transformer.py"
        ).read_text(encoding="utf-8")
        latent_source = (
            REPO_ROOT
            / "third_party"
            / "baselines"
            / "latent-code-inpainting"
            / "core"
            / "modules"
            / "transformer"
            / "mingpt.py"
        ).read_text(encoding="utf-8")

        self.assertIn("PUT_GA_SPG_LITE_LAMBDA", put_source)
        self.assertIn("PUT_GA_SPG_LITE_W_TEXTURE", put_source)
        self.assertIn("LATENT_SBVC_MODE", latent_source)
        self.assertIn("LATENT_SBVC_LAYER_IDS", latent_source)

    def test_result_table_contains_both_hosts(self):
        path = REPO_ROOT / "results" / "paper_main_results.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({"PUT", "Latent Codes"}, {row["backbone"] for row in rows})
        self.assertEqual(5, len(rows))


if __name__ == "__main__":
    unittest.main()
