"""Offline regression tests for resumable uploads and payload streaming."""
# Checkpoint restoration is exercised at its internal, network-free test seam.
# pylint: disable=protected-access

from __future__ import annotations

import io
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from aiohttp import ClientSession, web
from aiohttp.abc import AbstractStreamWriter
from aiohttp.payload import PAYLOAD_REGISTRY
from aiohttp.test_utils import TestServer
from oss2.exceptions import AccessDenied, NoSuchUpload
from oss2.models import PartInfo
from oss2.resumable import ResumableStore
from oss2.utils import _CHUNK_SIZE, Crc64

from aiooss2.adapter import AsyncPayload
from aiooss2.resumable import ResumableUploader
from aiooss2.utils import make_adapter


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    """Check checkpoint restoration against a mocked async bucket."""

    def setUp(self) -> None:
        # unittest cleanup keeps the directory alive for each async test.
        # pylint: disable-next=consider-using-with
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "data"
        self.path.write_bytes(b"abcdefgh")
        self.bucket = SimpleNamespace(
            bucket_name="offline-test",
            init_multipart_upload=AsyncMock(
                return_value=SimpleNamespace(upload_id="new")
            ),
            list_parts=AsyncMock(
                return_value=SimpleNamespace(
                    parts=[], is_truncated=False, next_marker="0"
                )
            ),
        )
        self.uploader = ResumableUploader(
            self.bucket,
            "key",
            str(self.path),
            8,
            store=ResumableStore(root=self.tmp.name),
            part_size=4,
            headers={
                "x-oss-request-payer": "requester",
                "x-oss-server-side-encryption": "AES256",
            },
        )

    async def seed(self) -> dict[str, Any]:
        """Persist a real checkpoint before simulating a resumed upload."""
        record = await self.uploader.init_record()
        self.bucket.init_multipart_upload.reset_mock()
        return record

    async def test_resume_keeps_upload_and_completed_parts(self) -> None:
        """Reuse the upload ID and completed parts with filtered headers."""
        await self.seed()
        part = PartInfo(1, "etag", size=4, part_crc=0)
        self.bucket.list_parts.side_effect = lambda *args, **kwargs: SimpleNamespace(
            parts=[part], is_truncated=False, next_marker="0"
        )
        await self.uploader._load_record()
        self.bucket.init_multipart_upload.assert_not_awaited()
        self.assertEqual(self.uploader._ResumableUploader__finished_size, 4)
        self.assertEqual(self.uploader._ResumableUploader__finished_parts, [part])
        headers = self.bucket.list_parts.call_args.kwargs["headers"]
        self.assertEqual(headers["x-oss-request-payer"], "requester")
        self.assertNotIn("x-oss-server-side-encryption", headers)

    async def test_empty_existing_upload_is_valid(self) -> None:
        """Keep an existing upload even when it has no completed parts."""
        await self.seed()
        await self.uploader._load_record()
        self.bucket.init_multipart_upload.assert_not_awaited()

    async def test_missing_upload_restarts(self) -> None:
        """Start a new upload when OSS reports NoSuchUpload."""
        await self.seed()
        self.bucket.list_parts.side_effect = [
            NoSuchUpload(404, {}, "", {}),
            SimpleNamespace(parts=[], is_truncated=False, next_marker="0"),
        ]
        await self.uploader._load_record()
        self.bucket.init_multipart_upload.assert_awaited_once()

    async def test_access_denied_preserves_record(self) -> None:
        """Propagate authorization errors without discarding the checkpoint."""
        record = await self.seed()
        self.bucket.list_parts.side_effect = AccessDenied(403, {}, "", {})
        with self.assertRaises(AccessDenied):
            await self.uploader._load_record()
        self.assertEqual(self.uploader._get_record(), record)
        self.bucket.init_multipart_upload.assert_not_awaited()

    async def test_new_upload(self) -> None:
        """Initialize an upload when no checkpoint exists."""
        await self.uploader._load_record()
        self.bucket.init_multipart_upload.assert_awaited_once()

    async def test_changed_file_restarts(self) -> None:
        """Discard a checkpoint for a changed file."""
        record = await self.seed()
        record["size"] = 9
        self.uploader._put_record(record)
        await self.uploader._load_record()
        self.bucket.init_multipart_upload.assert_awaited_once()

    async def test_invalid_record_restarts(self) -> None:
        """Discard malformed checkpoint data."""
        await self.seed()
        self.uploader._put_record({"op_type": "invalid"})
        await self.uploader._load_record()
        self.bucket.init_multipart_upload.assert_awaited_once()


class PayloadTests(unittest.IsolatedAsyncioTestCase):
    """Check upload contents and bounded reads without OSS credentials."""

    @staticmethod
    def record_progress(progress: list[int], consumed: int, _total: int) -> None:
        """Record progress for one adapter without closing over loop state."""
        progress.append(consumed)

    async def test_registered_payload_uploads_over_http(self) -> None:
        """Send both adapter types through the real HTTP transport."""
        body = b"payload" * 20000
        received: list[bytes] = []

        async def receive(request: web.Request) -> web.Response:
            received.append(await request.read())
            return web.Response(status=200)

        app = web.Application()
        app.router.add_put("/object", receive)
        async with TestServer(app) as server:
            async with ClientSession() as session:
                for stream in (body, io.BytesIO(body)):
                    adapter = make_adapter(stream, enable_crc=True)
                    payload = PAYLOAD_REGISTRY.get(adapter)
                    self.assertIsInstance(payload, AsyncPayload)
                    async with session.put(
                        server.make_url("/object"), data=adapter
                    ) as response:
                        self.assertEqual(response.status, 200)
        self.assertEqual(received, [body, body])

    async def test_bounded_writes_preserve_body_crc_and_progress(self) -> None:
        """Bound each write while preserving contents, CRC, and progress."""
        body = b"x" * (2 * 1024 * 1024 + 17)
        for stream in (body, io.BytesIO(body)):
            with self.subTest(stream_type=type(stream).__name__):
                progress: list[int] = []
                adapter = make_adapter(
                    stream,
                    enable_crc=True,
                    progress_callback=partial(self.record_progress, progress),
                )
                writer = AsyncMock(spec=AbstractStreamWriter)
                await AsyncPayload(adapter).write(writer)
                chunks = [call.args[0] for call in writer.write.await_args_list]
                self.assertEqual(b"".join(chunks), body)
                self.assertLessEqual(max(map(len, chunks)), _CHUNK_SIZE)
                expected_crc = Crc64()
                expected_crc(body)
                self.assertEqual(adapter.crc, expected_crc.crc)
                self.assertEqual(progress[-1], len(body))


if __name__ == "__main__":
    unittest.main(verbosity=2)
