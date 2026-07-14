from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from io import BytesIO
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
from PIL import Image


class UnsafeImageURL(ValueError):
    """Raised when an image URL points at a disallowed network location."""


class ImageDownloadError(RuntimeError):
    """Raised when an allowed image URL cannot be downloaded safely."""


@dataclass(frozen=True, slots=True)
class ImageDownloadLimits:
    timeout_seconds: float = 10.0
    max_redirects: int = 5
    max_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 25_000_000


DEFAULT_LIMITS = ImageDownloadLimits()
_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}


def _is_disallowed_ip(ip: ipaddress._BaseAddress) -> bool:
    return any(
        (
            ip.is_loopback,
            ip.is_private,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


def _iter_resolved_ips(hostname: str) -> Iterable[ipaddress._BaseAddress]:
    try:
        literal = ipaddress.ip_address(hostname)
        yield literal
        return
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeImageURL(f"Could not resolve host: {hostname}") from exc

    seen: set[str] = set()
    for info in infos:
        address = info[4][0]
        if address in seen:
            continue
        seen.add(address)
        yield ipaddress.ip_address(address)


def validate_public_image_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeImageURL("Only http and https image URLs are allowed")
    if not parsed.hostname:
        raise UnsafeImageURL("Image URL must include a hostname")

    hostname = parsed.hostname.strip().lower().rstrip(".")
    if hostname in _BLOCKED_HOSTNAMES:
        raise UnsafeImageURL("Local and metadata hostnames are not allowed")

    for ip in _iter_resolved_ips(hostname):
        if _is_disallowed_ip(ip):
            raise UnsafeImageURL(f"Disallowed network address: {ip}")

    return parsed.geturl()


async def validate_public_image_url_async(url: str) -> str:
    return await asyncio.to_thread(validate_public_image_url, url)


def validate_image_payload(image_bytes: bytes, limits: ImageDownloadLimits = DEFAULT_LIMITS) -> None:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            image.verify()
    except Exception as exc:
        raise ImageDownloadError("Downloaded payload is not a valid image") from exc

    if width * height > limits.max_pixels:
        raise ImageDownloadError("Downloaded image exceeds the maximum pixel limit")


async def download_public_image(
    url: str,
    limits: ImageDownloadLimits = DEFAULT_LIMITS,
) -> bytes:
    current_url = await validate_public_image_url_async(url)
    timeout = httpx.Timeout(limits.timeout_seconds)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for _ in range(limits.max_redirects + 1):
            async with client.stream("GET", current_url) as response:
                if 300 <= response.status_code < 400:
                    location = response.headers.get("location")
                    if not location:
                        raise ImageDownloadError("Redirect response missing Location header")
                    current_url = await validate_public_image_url_async(
                        urljoin(current_url, location)
                    )
                    continue

                response.raise_for_status()
                _validate_response_headers(response, limits)

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > limits.max_bytes:
                        raise ImageDownloadError("Downloaded image exceeds the maximum byte limit")
                    chunks.append(chunk)

                image_bytes = b"".join(chunks)
                await asyncio.to_thread(validate_image_payload, image_bytes, limits)
                return image_bytes

        raise ImageDownloadError("Too many redirects while downloading image")


def download_public_image_sync(
    url: str,
    limits: ImageDownloadLimits = DEFAULT_LIMITS,
) -> bytes:
    current_url = validate_public_image_url(url)
    timeout = httpx.Timeout(limits.timeout_seconds)

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for _ in range(limits.max_redirects + 1):
            with client.stream("GET", current_url) as response:
                if 300 <= response.status_code < 400:
                    location = response.headers.get("location")
                    if not location:
                        raise ImageDownloadError("Redirect response missing Location header")
                    current_url = validate_public_image_url(urljoin(current_url, location))
                    continue

                response.raise_for_status()
                _validate_response_headers(response, limits)

                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > limits.max_bytes:
                        raise ImageDownloadError("Downloaded image exceeds the maximum byte limit")
                    chunks.append(chunk)

                image_bytes = b"".join(chunks)
                validate_image_payload(image_bytes, limits)
                return image_bytes

        raise ImageDownloadError("Too many redirects while downloading image")


def _validate_response_headers(
    response: httpx.Response,
    limits: ImageDownloadLimits,
) -> None:
    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    if not content_type.startswith("image/"):
        raise ImageDownloadError("URL did not return an image content type")

    content_length = response.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > limits.max_bytes:
                raise ImageDownloadError("Image response exceeds the maximum byte limit")
        except ValueError:
            raise ImageDownloadError("Invalid Content-Length header")
