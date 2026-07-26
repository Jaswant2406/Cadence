from __future__ import annotations

import asyncio
import unittest

from fastapi import HTTPException, UploadFile

from cadence_api.main import create_run, csv_template, health, import_csv
from cadence_api.models import RunCreate


class ApiContractTests(unittest.TestCase):
    def test_health_contract(self) -> None:
        response = health()
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["service"], "cadence-api")

    def test_template_contract(self) -> None:
        response = csv_template()
        self.assertEqual(response.media_type, "text/csv")
        self.assertIn("cadence-template.csv", response.headers["content-disposition"])

    def test_non_csv_upload_is_rejected(self) -> None:
        upload = UploadFile(filename="events.txt", file=None)
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(import_csv(upload))
        self.assertEqual(raised.exception.status_code, 415)

    def test_csv_run_requires_import_id(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            create_run(RunCreate(source="csv"))
        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
