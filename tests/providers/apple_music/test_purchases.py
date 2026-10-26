"""Unit tests for promoting Apple Music purchases that Apple still sells."""

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.errors import MediaNotFoundError, RetriesExhausted

from music_assistant.providers.apple_music.media import (
    AppleMusicMediaManager,
    get_purchases_still_sold,
)
from tests.common import use_real_create_task

if TYPE_CHECKING:
    from music_assistant_models.media_items import Track


def _catalog_song(catalog_id: str) -> dict[str, Any]:
    """Build a minimal catalog/songs response item."""
    return {
        "id": catalog_id,
        "type": "songs",
        "attributes": {
            "name": f"Catalog {catalog_id}",
            "artistName": "Artist",
            "albumName": "Album",
            "durationInMillis": 180000,
            "playParams": {"id": catalog_id},
        },
        "relationships": {},
    }


def _library_song(library_id: str, *, catalog_id: str | None = None) -> dict[str, Any]:
    """Build a me/library/songs response item, optionally carrying its catalog twin."""
    relationships: dict[str, Any] = {}
    if catalog_id is not None:
        relationships["catalog"] = {"data": [_catalog_song(catalog_id)]}
    return {
        "id": library_id,
        "type": "library-songs",
        "attributes": {
            "name": f"Library {library_id}",
            "artistName": "Artist",
            "albumName": "Album",
            "durationInMillis": 180000,
            "playParams": {"id": library_id},
        },
        "relationships": relationships,
    }


@pytest.fixture
def mock_api() -> MagicMock:
    """Return a MagicMock representing the Apple Music API client."""
    api_client = MagicMock()
    api_client.get_data = AsyncMock()
    return api_client


@pytest.fixture
def manager(mock_api: MagicMock) -> AppleMusicMediaManager:
    """Return an AppleMusicMediaManager wired to a mock API client and a always-miss cache."""
    provider = MagicMock()
    provider.instance_id = "apple_music_test"
    provider.domain = "apple_music"
    provider._storefront = "us"
    provider.logger = MagicMock()
    provider.mass.cache.get = AsyncMock(return_value=None)
    provider.mass.cache.get_with_freshness = AsyncMock(return_value=(None, False, False))
    provider.mass.cache.set = AsyncMock()
    use_real_create_task(provider.mass)
    provider.api_client = mock_api
    manager = AppleMusicMediaManager(provider)
    provider.media_manager = manager
    return manager


def _purchase(library_id: str, purchased_id: str) -> dict[str, Any]:
    """Build a purchased library song: a purchasedId and no catalogId."""
    item = _library_song(library_id)
    item["attributes"]["playParams"]["purchasedId"] = purchased_id
    return item


def _availability(track: Track) -> list[bool]:
    return [mapping.available for mapping in track.provider_mappings]


@pytest.mark.asyncio
async def test_purchases_still_sold_asks_the_catalog_once_for_purchases_only(
    manager: AppleMusicMediaManager,
    mock_api: MagicMock,
) -> None:
    """Only purchases are looked up, in one request, and only the ones found are returned."""
    mock_api.get_data.return_value = {"data": [{"id": "397010985"}]}
    items = [
        _purchase("i.sold", "397010985"),
        _purchase("i.revoked", "915794195"),
        _library_song("i.upload"),
    ]

    assert await get_purchases_still_sold(manager.provider, items) == {"i.sold"}
    mock_api.get_data.assert_awaited_once_with("catalog/us/songs", ids="397010985,915794195")


@pytest.mark.asyncio
async def test_purchases_still_sold_treats_a_404_as_none_sold(
    manager: AppleMusicMediaManager,
    mock_api: MagicMock,
) -> None:
    """Apple answers 404 when it recognises none of the ids; that is an answer, not a failure."""
    mock_api.get_data.side_effect = MediaNotFoundError("not found")

    assert (
        await get_purchases_still_sold(manager.provider, [_purchase("i.u2", "915794195")]) == set()
    )


@pytest.mark.asyncio
async def test_purchases_still_sold_hides_purchases_when_throttled(
    manager: AppleMusicMediaManager,
    mock_api: MagicMock,
) -> None:
    """A throttled lookup leaves purchases hidden instead of raising."""
    mock_api.get_data.side_effect = RetriesExhausted("throttled")

    assert (
        await get_purchases_still_sold(manager.provider, [_purchase("i.sold", "397010985")])
        == set()
    )


@pytest.mark.asyncio
async def test_purchases_still_sold_makes_no_request_without_purchases(
    manager: AppleMusicMediaManager,
    mock_api: MagicMock,
) -> None:
    """A listing without purchases, the common case, costs nothing."""
    assert await get_purchases_still_sold(manager.provider, [_library_song("i.upload")]) == set()
    mock_api.get_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_playlist_tracks_promote_purchases_apple_still_sells(
    manager: AppleMusicMediaManager,
    mock_api: MagicMock,
) -> None:
    """Playlist tracks are parsed fresh, so they need the promotion too or the queue skips them."""

    async def _get_data(endpoint: str, **_kwargs: Any) -> dict[str, Any]:
        if endpoint == "catalog/us/songs":
            return {"data": [{"id": "397010985"}]}
        return {"data": [_purchase("i.sold", "397010985"), _purchase("i.revoked", "915794195")]}

    mock_api.get_data.side_effect = _get_data

    tracks = await manager._get_playlist_tracks_cached("p.purchased", 0, None)

    assert [_availability(track) for track in tracks] == [[True], [False]]


@pytest.mark.asyncio
async def test_library_album_tracks_promote_purchases_apple_still_sells(
    manager: AppleMusicMediaManager,
    mock_api: MagicMock,
) -> None:
    """The tracks of a purchased library album get the same promotion."""

    async def _get_data(endpoint: str, **_kwargs: Any) -> dict[str, Any]:
        if endpoint == "catalog/us/songs":
            return {"data": [{"id": "397010985"}]}
        return {"data": [_purchase("i.sold", "397010985"), _purchase("i.revoked", "915794195")]}

    mock_api.get_data.side_effect = _get_data

    tracks = await manager.get_album_tracks("l.ZGZGdjd")

    assert [_availability(track) for track in tracks] == [[True], [False]]
