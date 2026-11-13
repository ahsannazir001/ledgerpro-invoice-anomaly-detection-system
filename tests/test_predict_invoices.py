import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_invoices import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    FROZEN_THRESHOLD,
    load_config,
    predict_records,
    predict_with_pipeline,
    prepare_inference_features,
    resolve_model_artifact,
    validate_output_path,
    write_predictions_atomically,
)


def valid_records(count=1):
    return pd.DataFrame({
        "invoice_id": [f"INV-{index:03d}" for index in range(count)],
        "amount": [1000.0] * count,
        "payment_terms_days": [30] * count,
        "line_item_count": [2] * count,
        "invoice_date": ["2025-01-01"] * count,
        "due_date": ["2025-01-31"] * count,
    })


class PredictInvoicesTests(unittest.TestCase):
    def test_frozen_threshold_uses_predict_proba_and_class_1(self):
        config = load_config()
        self.assertEqual(config["threshold"], FROZEN_THRESHOLD)

        pipeline = Mock()
        pipeline.classes_ = np.array([0, 1])
        pipeline.predict_proba.return_value = np.array([
            [0.421, 0.579],
            [0.420, 0.580],
            [0.100, 0.900],
        ])
        results = predict_with_pipeline(valid_records(3), pipeline, 0.58)

        pipeline.predict_proba.assert_called_once()
        pipeline.predict.assert_not_called()
        self.assertEqual(results["is_fraud_prediction"].tolist(), [0, 1, 1])
        self.assertTrue((results["threshold"] == FROZEN_THRESHOLD).all())

    def test_rejects_unexpected_class_labels_or_order(self):
        for classes in ([1, 0], [0, 2], [0, 1, 2]):
            with self.subTest(classes=classes):
                pipeline = Mock()
                pipeline.classes_ = np.asarray(classes)
                with self.assertRaisesRegex(ValueError, "classes_ must be exactly"):
                    predict_with_pipeline(valid_records(), pipeline, FROZEN_THRESHOLD)
                pipeline.predict_proba.assert_not_called()

    def test_rejects_any_threshold_other_than_frozen_value(self):
        pipeline = Mock()
        pipeline.classes_ = np.array([0, 1])
        with self.assertRaisesRegex(ValueError, "frozen at 0.58"):
            predict_with_pipeline(valid_records(), pipeline, 0.5)
        pipeline.predict_proba.assert_not_called()

    def test_reproduces_saved_evaluation_predictions_and_probabilities(self):
        evaluation_data = pd.read_csv(
            ROOT / "data" / "external_test"
            / "threshold_optimized_external_seed_20261112.csv"
        )
        saved_predictions = pd.read_csv(
            ROOT / "reports" / "threshold_optimized_external_predictions.csv"
        )

        inference_results = predict_records(evaluation_data)

        self.assertEqual(
            inference_results["invoice_id"].tolist(),
            saved_predictions["invoice_id"].tolist(),
        )
        self.assertEqual(
            inference_results["is_fraud_prediction"].tolist(),
            saved_predictions["predicted"].tolist(),
        )
        np.testing.assert_allclose(
            inference_results["fraud_probability"].to_numpy(),
            saved_predictions["fraud_probability"].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "Missing required input columns"):
            prepare_inference_features(valid_records().drop(columns=["amount"]))

    def test_rejects_duplicate_invoice_ids(self):
        records = valid_records(2)
        records.loc[1, "invoice_id"] = records.loc[0, "invoice_id"]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            prepare_inference_features(records)

    def test_rejects_invalid_dates(self):
        records = valid_records()
        records.loc[0, "invoice_date"] = "not-a-date"
        with self.assertRaisesRegex(ValueError, "invoice_date must contain valid"):
            prepare_inference_features(records)

    def test_rejects_nan_and_infinity(self):
        for amount in (np.nan, np.inf, -np.inf):
            with self.subTest(amount=amount):
                records = valid_records()
                records.loc[0, "amount"] = amount
                with self.assertRaises(ValueError):
                    prepare_inference_features(records)

    def test_rejects_negative_and_out_of_range_amounts(self):
        for amount in (-1, 49.99, 250000.01):
            with self.subTest(amount=amount):
                records = valid_records()
                records.loc[0, "amount"] = amount
                with self.assertRaisesRegex(ValueError, "amount must be between"):
                    prepare_inference_features(records)

    def test_rejects_invalid_payment_terms(self):
        for terms in (0, 20, 90, 30.5):
            with self.subTest(terms=terms):
                records = valid_records()
                records["payment_terms_days"] = records["payment_terms_days"].astype(float)
                records.loc[0, "payment_terms_days"] = terms
                with self.assertRaisesRegex(ValueError, "payment_terms_days must be one"):
                    prepare_inference_features(records)

    def test_rejects_invalid_line_item_counts(self):
        for count in (0, 16, 2.5, np.inf):
            with self.subTest(count=count):
                records = valid_records()
                records["line_item_count"] = records["line_item_count"].astype(float)
                records.loc[0, "line_item_count"] = count
                with self.assertRaises(ValueError):
                    prepare_inference_features(records)

    def test_rejects_unknown_columns(self):
        records = valid_records()
        records["fraud_hint"] = 1
        with self.assertRaisesRegex(ValueError, "Unexpected input columns"):
            prepare_inference_features(records)

    def test_rejects_malformed_or_changed_config(self):
        original = load_config()
        bad_configs = []

        missing = copy.deepcopy(original)
        del missing["threshold"]
        bad_configs.append(missing)

        wrong_type = copy.deepcopy(original)
        wrong_type["threshold"] = "0.58"
        bad_configs.append(wrong_type)

        changed_threshold = copy.deepcopy(original)
        changed_threshold["threshold"] = 0.59
        bad_configs.append(changed_threshold)

        bad_classifier = copy.deepcopy(original)
        bad_classifier["classifier"]["parameters"]["n_estimators"] = "200"
        bad_configs.append(bad_classifier)

        for config in bad_configs:
            with self.subTest(config=config):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "config.json"
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_rejects_path_traversal_absolute_and_unapproved_artifacts(self):
        config = load_config()
        for artifact in (
            "models/../README.md",
            str((ROOT / "models" / "threshold_optimized_pipeline.joblib").resolve()),
            "models/final_pipeline.joblib",
        ):
            with self.subTest(artifact=artifact):
                altered = dict(config, artifact=artifact)
                with self.assertRaises(ValueError):
                    resolve_model_artifact(altered)

    def test_output_path_protects_input_and_existing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "invoices.csv"
            output_path = Path(temp_dir) / "results.csv"
            input_path.write_text("source", encoding="utf-8")
            output_path.write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cannot overwrite the input"):
                write_predictions_atomically(
                    pd.DataFrame({"prediction": [1]}), input_path, input_path
                )
            with self.assertRaises(FileExistsError):
                write_predictions_atomically(
                    pd.DataFrame({"prediction": [1]}), input_path, output_path
                )
            self.assertEqual(output_path.read_text(encoding="utf-8"), "existing")
            self.assertEqual(input_path.read_text(encoding="utf-8"), "source")

            written_path = write_predictions_atomically(
                pd.DataFrame({"prediction": [1]}),
                input_path,
                output_path,
                overwrite=True,
            )
            self.assertEqual(written_path, output_path.resolve())
            self.assertEqual(
                pd.read_csv(output_path)["prediction"].tolist(), [1]
            )

    def test_successful_output_is_closed_before_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "invoices.csv"
            output_path = Path(temp_dir) / "results.csv"
            input_path.write_text("source", encoding="utf-8")
            predictions = pd.DataFrame({"invoice_id": ["A", "B"], "prediction": [0, 1]})
            real_link = os.link
            publish_observations = []

            def inspect_then_link(source, destination):
                source_path = Path(source)
                self.assertEqual(source_path.parent, output_path.parent)
                self.assertEqual(
                    pd.read_csv(source_path).to_dict("list"),
                    predictions.to_dict("list"),
                )
                publish_observations.append(source_path)
                return real_link(source, destination)

            with patch("predict_invoices.os.link", side_effect=inspect_then_link):
                result = write_predictions_atomically(
                    predictions, input_path, output_path
                )

            self.assertEqual(result, output_path.resolve())
            self.assertEqual(len(publish_observations), 1)
            self.assertEqual(
                pd.read_csv(output_path).to_dict("list"),
                predictions.to_dict("list"),
            )
            self.assertEqual(list(Path(temp_dir).glob(".results.csv.*.tmp")), [])

    def test_write_failure_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "invoices.csv"
            output_path = Path(temp_dir) / "results.csv"
            input_path.write_text("source", encoding="utf-8")
            with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "Could not safely publish"):
                    write_predictions_atomically(
                        pd.DataFrame({"prediction": [1]}), input_path, output_path
                    )
            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(temp_dir).glob(".results.csv.*.tmp")), [])

    def test_publish_failure_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "invoices.csv"
            output_path = Path(temp_dir) / "results.csv"
            input_path.write_text("source", encoding="utf-8")
            with patch("predict_invoices.os.link", side_effect=OSError("publish failed")):
                with self.assertRaisesRegex(OSError, "Could not safely publish"):
                    write_predictions_atomically(
                        pd.DataFrame({"prediction": [1]}), input_path, output_path
                    )
            self.assertFalse(output_path.exists())
            self.assertEqual(list(Path(temp_dir).glob(".results.csv.*.tmp")), [])

    def test_default_and_overwrite_output_path_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "invoices.csv"
            output_path = Path(temp_dir) / "results.csv"
            input_path.write_text("source", encoding="utf-8")
            output_path.write_text("existing", encoding="utf-8")
            self.assertEqual(input_path.read_text(encoding="utf-8"), "source")
            with self.assertRaises(FileExistsError):
                validate_output_path(input_path, output_path, overwrite=False)
            self.assertEqual(
                validate_output_path(input_path, output_path, overwrite=True),
                output_path.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
