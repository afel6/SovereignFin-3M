"""The local services file publishes pinned images on the loopback address only."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.yml"
LOOPBACK_PREFIX = "127.0.0.1:"
POSTGRES_PORT = "5432"
QDRANT_PORT = "6333"
POSTGRES_IMAGE = re.compile(r"postgres:16\.\d+")
QDRANT_IMAGE = re.compile(r"qdrant/qdrant:v\d+\.\d+\.\d+")
NAMED_VOLUME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]+")
SERVICE_KEY = re.compile(r"  ([A-Za-z0-9_-]+):\s*")
QUOTES = "\"'"


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in QUOTES:
        return value[1:-1]
    return value


def service_blocks(text: str) -> dict[str, list[str]]:
    """Split the lines under `services:` by their 2-space service keys."""
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    inside = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if indent_of(line) == 0:
            inside = line.rstrip() == "services:"
            current = None
            continue
        if not inside:
            continue
        match = SERVICE_KEY.fullmatch(line)
        if match:
            current = blocks.setdefault(match.group(1), [])
        elif current is not None:
            current.append(line)
    return blocks


def _key_body(lines: list[str], key: str) -> list[str]:
    """Lines nested under `key:`; fails closed on flow style, a missing key or no entries."""
    for pos, line in enumerate(lines):
        if line.strip().startswith(f"{key}:"):
            if line.strip() != f"{key}:":
                raise ValueError(f"{key} must be block style")
            body = []
            for nxt in lines[pos + 1 :]:
                if indent_of(nxt) <= indent_of(line):
                    break
                body.append(nxt)
            if not body:
                raise ValueError(f"{key} has no entries")
            return body
    raise ValueError(f"{key} not found")


def list_items(lines: list[str], key: str) -> list[str]:
    """The `- ` items under `key:` with surrounding quotes stripped."""
    items = [ln.strip()[2:] for ln in _key_body(lines, key) if ln.strip().startswith("- ")]
    if not items:
        raise ValueError(f"{key} has no list items")
    return [unquote(item) for item in items]


def map_items(lines: list[str], key: str) -> dict[str, str]:
    """The `NAME: value` entries under `key:` with surrounding quotes stripped."""
    result = {}
    for line in _key_body(lines, key):
        name, _, value = line.strip().partition(":")
        result[name] = unquote(value)
    return result


def image_of(lines: list[str]) -> str:
    return next(unquote(ln.strip()[6:]) for ln in lines if ln.strip().startswith("image:"))


def is_loopback_port(item: str) -> bool:
    return item.startswith(LOOPBACK_PREFIX)


def container_port(item: str) -> str:
    return item.rsplit(":", 1)[1].split("/")[0]


def is_named_volume(item: str) -> bool:
    return NAMED_VOLUME.fullmatch(item.split(":")[0]) is not None


@pytest.fixture(scope="module")
def text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def services(text) -> dict[str, list[str]]:
    return service_blocks(text)


SAMPLE = """\
name: x
services:
  a:
    image: "i:1"
    ports:
      - "127.0.0.1:1:1"
      - '2:2'
    volumes:
      - data:/d
  b:
    ports: ["1:1"]
volumes:
  data:
"""


def test_reader_splits_services_and_stops_at_top_level():
    """A reader that ran past `services:` would treat top-level volumes as a service."""
    assert list(service_blocks(SAMPLE)) == ["a", "b"]


def test_reader_returns_items_without_quotes():
    """A reader that kept the quote characters would break the prefix check."""
    assert list_items(service_blocks(SAMPLE)["a"], "ports") == ["127.0.0.1:1:1", "2:2"]


def test_reader_fails_closed_on_flow_style_and_missing_key():
    """A reader that returned nothing for flow style would let unbound ports pass."""
    block = service_blocks(SAMPLE)["b"]
    with pytest.raises(ValueError):
        list_items(block, "ports")
    with pytest.raises(ValueError):
        list_items(block, "volumes")


def test_reader_fails_closed_on_empty_key():
    """A `ports:` key with no items must not read as an empty, passing list."""
    with pytest.raises(ValueError):
        list_items(["    ports:", "    image: x"], "ports")


@pytest.mark.parametrize("item", ["5432:5432", "0.0.0.0:5432:5432"])
def test_loopback_check_rejects_unbound_ports(item):
    """A check that accepted a bare or all-interface port would expose the service."""
    assert not is_loopback_port(item)


def test_named_volume_check_rejects_host_paths():
    """A check that accepted host paths would bind-mount the user's files."""
    assert is_named_volume("pgdata:/var/lib/x")
    assert not is_named_volume("./data:/x")
    assert not is_named_volume("/abs:/x")
    assert not is_named_volume("C:\\data:/x")


def test_services_are_postgres_and_qdrant(services):
    """A file missing either service leaves the stack incomplete."""
    assert {"postgres", "qdrant"} <= set(services)


def test_postgres_image_is_pinned_minor(services):
    """A `latest` or bare `16` tag would change the database under the user."""
    assert POSTGRES_IMAGE.fullmatch(image_of(services["postgres"]))


def test_qdrant_image_is_pinned(services):
    """A floating qdrant tag would change the vector store without notice."""
    assert QDRANT_IMAGE.fullmatch(image_of(services["qdrant"]))


@pytest.mark.parametrize("name", ["postgres", "qdrant"])
def test_ports_bound_to_loopback_only(services, name):
    """A port published on all interfaces would expose the database to the network."""
    ports = list_items(services[name], "ports")
    assert all(is_loopback_port(item) for item in ports)


@pytest.mark.parametrize(("name", "port"), [("postgres", POSTGRES_PORT), ("qdrant", QDRANT_PORT)])
def test_published_container_port(services, name, port):
    """Publishing the wrong container port leaves the service unreachable."""
    assert [container_port(i) for i in list_items(services[name], "ports")] == [port]


def test_qdrant_telemetry_disabled(services):
    """Telemetry left on would send usage data off the machine."""
    env = map_items(services["qdrant"], "environment")
    assert env["QDRANT__TELEMETRY_DISABLED"] == "true"


def test_qdrant_cors_disabled(services):
    """Qdrant's default CORS lets any web page in the user's browser read or delete the index."""
    env = map_items(services["qdrant"], "environment")
    assert env["QDRANT__SERVICE__ENABLE_CORS"] == "false"


def test_postgres_password_comes_from_env_file(services):
    """A literal password in the file would be committed with it."""
    env = map_items(services["postgres"], "environment")
    assert env["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD")


@pytest.mark.parametrize("name", ["postgres", "qdrant"])
def test_volumes_are_named(services, name):
    """A host-path mount would tie data to one machine's folders."""
    assert all(is_named_volume(i) for i in list_items(services[name], "volumes"))


def test_no_init_scripts_mounted(text):
    """A mounted init script would run unreviewed SQL on first start."""
    assert "docker-entrypoint-initdb.d" not in text
