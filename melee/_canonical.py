"""Raw byte -> canonical dataclass readers shared by Console and SLPFileStreamer.

These mirror peppi's `Post::read_push` / `Pre::read_push` / item reader so that
`GameState.to_canonical_dict()` produces output byte-for-byte identical to
`peppi_py.read_frame_dicts`. Offsets are libmelee's event_bytes layout (which
includes the event command byte at index 0). slp-spec offsets = these - 0.

Version-gated fields use struct.unpack_from; out-of-range reads are caught and
the field stays None — never default-zero — so consumers can distinguish
"slp too old" from "field present, value is zero".
"""
from __future__ import annotations

import functools
import struct
from typing import Optional

from melee.gamestate import (
	CanonicalStadiumTransformation, DreamlandWhispy, FodPlatform,
	FrameEnd, FrameStart, Item, Position, Post, Pre, TriggersPhysical,
	Velocities, Velocity,
)


def _maybe(fmt: str, buf: bytes, off: int):
	try:
		return struct.unpack_from(fmt, buf, off)[0]
	except struct.error:
		return None


def read_post(buf: bytes, version: tuple[int, int, int]) -> Post:
	p = Post()
	p.character = struct.unpack_from('>B', buf, 0x07)[0]
	p.state = struct.unpack_from('>H', buf, 0x08)[0]
	p.position = Position(
		struct.unpack_from('>f', buf, 0x0A)[0],
		struct.unpack_from('>f', buf, 0x0E)[0],
	)
	p.direction = struct.unpack_from('>f', buf, 0x12)[0]
	p.percent = struct.unpack_from('>f', buf, 0x16)[0]
	p.shield = struct.unpack_from('>f', buf, 0x1A)[0]
	p.last_attack_landed = struct.unpack_from('>B', buf, 0x1E)[0]
	p.combo_count = struct.unpack_from('>B', buf, 0x1F)[0]
	p.last_hit_by = struct.unpack_from('>B', buf, 0x20)[0]
	p.stocks = struct.unpack_from('>B', buf, 0x21)[0]

	if version >= (0, 2, 0):
		p.state_age = _maybe('>f', buf, 0x22)
	if version >= (2, 0, 0):
		flags = struct.unpack_from('>BBBBB', buf, 0x26) if len(buf) >= 0x2B else None
		p.state_flags = flags
		p.misc_as = _maybe('>f', buf, 0x2B)
		p.airborne = _maybe('>B', buf, 0x2F)
		p.ground = _maybe('>H', buf, 0x30)
		p.jumps = _maybe('>B', buf, 0x32)
		p.l_cancel = _maybe('>B', buf, 0x33)
	if version >= (2, 1, 0):
		p.hurtbox_state = _maybe('>B', buf, 0x34)
	if version >= (3, 5, 0) and len(buf) >= 0x49:
		v = struct.unpack_from('>fffff', buf, 0x35)
		p.velocities = Velocities(v[0], v[1], v[2], v[3], v[4])
	if version >= (3, 8, 0):
		p.hitlag = _maybe('>f', buf, 0x49)
	if version >= (3, 11, 0):
		p.animation_index = _maybe('>I', buf, 0x4D)
	if version >= (3, 16, 0):
		p.last_hit_by_instance = _maybe('>H', buf, 0x51)
		p.instance_id = _maybe('>H', buf, 0x53)
	return p


def read_pre(buf: bytes, version: tuple[int, int, int]) -> Pre:
	p = Pre()
	p.random_seed = struct.unpack_from('>I', buf, 0x07)[0]
	p.state = struct.unpack_from('>H', buf, 0x0B)[0]
	p.position = Position(
		struct.unpack_from('>f', buf, 0x0D)[0],
		struct.unpack_from('>f', buf, 0x11)[0],
	)
	p.direction = struct.unpack_from('>f', buf, 0x15)[0]
	p.joystick = Position(
		struct.unpack_from('>f', buf, 0x19)[0],
		struct.unpack_from('>f', buf, 0x1D)[0],
	)
	p.cstick = Position(
		struct.unpack_from('>f', buf, 0x21)[0],
		struct.unpack_from('>f', buf, 0x25)[0],
	)
	p.triggers = struct.unpack_from('>f', buf, 0x29)[0]
	p.buttons = struct.unpack_from('>I', buf, 0x2D)[0]
	p.buttons_physical = struct.unpack_from('>H', buf, 0x31)[0]
	p.triggers_physical = TriggersPhysical(
		struct.unpack_from('>f', buf, 0x33)[0],
		struct.unpack_from('>f', buf, 0x37)[0],
	)
	if version >= (1, 2, 0):
		p.raw_analog_x = _maybe('>b', buf, 0x3B)
	if version >= (1, 4, 0):
		p.percent = _maybe('>f', buf, 0x3C)
	if version >= (3, 15, 0):
		p.raw_analog_y = _maybe('>b', buf, 0x40)
	if version >= (3, 17, 0):
		p.raw_analog_cstick_x = _maybe('>b', buf, 0x41)
		p.raw_analog_cstick_y = _maybe('>b', buf, 0x42)
	return p


def read_item(buf: bytes, version: tuple[int, int, int]) -> Item:
	it = Item()
	it.type = struct.unpack_from('>H', buf, 0x05)[0]
	it.state = struct.unpack_from('>B', buf, 0x07)[0]
	it.direction = struct.unpack_from('>f', buf, 0x08)[0]
	it.velocity = Velocity(
		struct.unpack_from('>f', buf, 0x0C)[0],
		struct.unpack_from('>f', buf, 0x10)[0],
	)
	it.position = Position(
		struct.unpack_from('>f', buf, 0x14)[0],
		struct.unpack_from('>f', buf, 0x18)[0],
	)
	it.damage = struct.unpack_from('>H', buf, 0x1C)[0]
	it.timer = struct.unpack_from('>f', buf, 0x1E)[0]
	it.id = struct.unpack_from('>I', buf, 0x22)[0]
	if version >= (3, 2, 0) and len(buf) >= 0x2A:
		it.misc = struct.unpack_from('>BBBB', buf, 0x26)
	if version >= (3, 6, 0):
		it.owner = _maybe('>b', buf, 0x2A)
	if version >= (3, 16, 0):
		it.instance_id = _maybe('>H', buf, 0x2B)
	return it


def read_frame_start(buf: bytes, version: tuple[int, int, int]) -> FrameStart:
	fs = FrameStart()
	fs.random_seed = struct.unpack_from('>I', buf, 0x05)[0]
	if version >= (3, 10, 0):
		fs.scene_frame_counter = _maybe('>I', buf, 0x09)
	return fs


def read_frame_end(buf: bytes, version: tuple[int, int, int]) -> FrameEnd:
	fe = FrameEnd()
	if version >= (3, 7, 0):
		fe.latest_finalized_frame = _maybe('>i', buf, 0x05)
	return fe


def read_fod_platform(buf: bytes, version: tuple[int, int, int]) -> FodPlatform:
	# slp 3.18+: per-frame FOD platform event payload
	# 0x05 platform u8, 0x06 height f32 BE
	return FodPlatform(
		platform=struct.unpack_from('>B', buf, 0x05)[0],
		height=struct.unpack_from('>f', buf, 0x06)[0],
	)


def read_dreamland_whispy(buf: bytes, version: tuple[int, int, int]) -> DreamlandWhispy:
	return DreamlandWhispy(direction=struct.unpack_from('>B', buf, 0x05)[0])


def read_stadium_transformation(buf: bytes, version: tuple[int, int, int]) -> CanonicalStadiumTransformation:
	return CanonicalStadiumTransformation(
		event=struct.unpack_from('>B', buf, 0x05)[0],
		type=struct.unpack_from('>H', buf, 0x07)[0],
	)


@functools.lru_cache(maxsize=8)
def event_list_names(version: tuple[int, int, int]) -> tuple[str, ...]:
	"""CanonicalFrame fields that should be reset to `[]` (not `None`) at
	FRAME_START for the given slp version.

	None vs [] is load-bearing: peppi emits `Option<Vec<…>>`, where None means
	"event type not present in this slp version" and [] means "present but no
	event this frame". The to_canonical_dict serializer must preserve that
	distinction. peppi's transpose::Frame declares all three stage-event Vecs
	unconditionally for slp >= 3.18, so e.g. a Pokémon Stadium frame still
	carries empty `fod_platforms` and `dreamland_whispys` lists.
	"""
	names = []
	if version >= (3, 0, 0):
		names.append('items')
	if version >= (3, 18, 0):
		names.extend(('fod_platforms', 'dreamland_whispys', 'stadium_transformations'))
	return tuple(names)
