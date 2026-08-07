#!/usr/bin/python3
"""Canonical-schema regression tests.

Pin the libmelee→peppi-py parity contract documented in README.md so that
`gamestate.to_canonical_dict()` stays byte-equivalent to
`peppi_py.read_frame_dicts(slp)[i]`. The peppi-parity test is skipped if
peppi_py is not installed; the other tests run standalone.
"""
import math
import struct
import unittest

import melee
from melee import _canonical
from melee.console import _finite_int_or_zero
from melee.gamestate import (
    CanonicalFrame, Data, FodPlatform, Item, PortData, Post, Pre,
    DreamlandWhispy, CanonicalStadiumTransformation, GameState,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers: build synthetic event byte buffers for the byte-reader unit
# tests. Each helper produces a buffer matching peppi's wire format for
# the corresponding event, with offsets relative to the libmelee
# event_bytes layout (which includes the command byte at index 0).
# ─────────────────────────────────────────────────────────────────────

def _post_bytes(version: tuple, **overrides) -> bytes:
    """Pad a POST_FRAME payload to whatever length matches `version`."""
    sizes = {
        (0, 1, 0): 0x22,
        (0, 2, 0): 0x26,
        (2, 0, 0): 0x34,
        (2, 1, 0): 0x35,
        (3, 5, 0): 0x4A,
        (3, 8, 0): 0x4E,
        (3, 11, 0): 0x52,
        (3, 16, 0): 0x56,
    }
    # Pick the largest defined size <= version.
    size = max(s for v, s in sizes.items() if v <= version)
    buf = bytearray(size)
    buf[0] = 0x38  # POST_FRAME command
    # Required fields (slp 0.1+).
    buf[0x07] = overrides.get('character', 2)             # u8
    struct.pack_into('>H', buf, 0x08, overrides.get('state', 14))
    struct.pack_into('>f', buf, 0x0A, overrides.get('pos_x', 1.5))
    struct.pack_into('>f', buf, 0x0E, overrides.get('pos_y', -2.5))
    struct.pack_into('>f', buf, 0x12, overrides.get('direction', 1.0))
    struct.pack_into('>f', buf, 0x16, overrides.get('percent', 0.0))
    struct.pack_into('>f', buf, 0x1A, overrides.get('shield', 60.0))
    if version >= (2, 0, 0):
        struct.pack_into('>f', buf, 0x22, overrides.get('state_age', 0.0))
        struct.pack_into('>BBBBB', buf, 0x26, *overrides.get('state_flags', (0, 0, 0, 0, 0)))
        struct.pack_into('>f', buf, 0x2B, overrides.get('misc_as', 0.0))
        buf[0x2F] = overrides.get('airborne', 0)          # raw u8
        struct.pack_into('>H', buf, 0x30, overrides.get('ground', 0))
        buf[0x32] = overrides.get('jumps', 1)
        buf[0x33] = overrides.get('l_cancel', 0)
    if version >= (2, 1, 0):
        buf[0x34] = overrides.get('hurtbox_state', 0)
    if version >= (3, 5, 0):
        struct.pack_into('>fffff', buf, 0x35, *overrides.get(
            'velocities', (0.0, 0.0, 0.0, 0.0, 0.0)))
    if version >= (3, 8, 0):
        struct.pack_into('>f', buf, 0x49 + 1, overrides.get('hitlag', 0.0))
    return bytes(buf)


def _pre_bytes(version: tuple, **overrides) -> bytes:
    sizes = {
        (0, 1, 0): 0x3B,
        (1, 2, 0): 0x3C,
        (1, 4, 0): 0x40,
        (3, 15, 0): 0x41,
        (3, 17, 0): 0x43,
    }
    size = max(s for v, s in sizes.items() if v <= version)
    buf = bytearray(size)
    buf[0] = 0x37  # PRE_FRAME
    struct.pack_into('>I', buf, 0x07, overrides.get('random_seed', 0))
    struct.pack_into('>H', buf, 0x0B, overrides.get('state', 0))
    struct.pack_into('>f', buf, 0x0D, overrides.get('pos_x', 0.0))
    struct.pack_into('>f', buf, 0x11, overrides.get('pos_y', 0.0))
    struct.pack_into('>f', buf, 0x15, overrides.get('direction', 1.0))
    struct.pack_into('>f', buf, 0x19, overrides.get('joy_x', 0.0))
    struct.pack_into('>f', buf, 0x1D, overrides.get('joy_y', 0.0))
    struct.pack_into('>f', buf, 0x21, overrides.get('cstick_x', 0.0))
    struct.pack_into('>f', buf, 0x25, overrides.get('cstick_y', 0.0))
    struct.pack_into('>f', buf, 0x29, overrides.get('triggers', 0.0))
    struct.pack_into('>I', buf, 0x2D, overrides.get('buttons', 0))
    struct.pack_into('>H', buf, 0x31, overrides.get('buttons_physical', 0))
    struct.pack_into('>f', buf, 0x33, overrides.get('trig_l', 0.0))
    struct.pack_into('>f', buf, 0x37, overrides.get('trig_r', 0.0))
    if version >= (1, 2, 0):
        struct.pack_into('>b', buf, 0x3B, overrides.get('raw_analog_x', 0))
    if version >= (1, 4, 0):
        struct.pack_into('>f', buf, 0x3C, overrides.get('percent', 0.0))
    if version >= (3, 15, 0):
        struct.pack_into('>b', buf, 0x40, overrides.get('raw_analog_y', 0))
    if version >= (3, 17, 0):
        struct.pack_into('>b', buf, 0x41, overrides.get('raw_analog_cx', 0))
        struct.pack_into('>b', buf, 0x42, overrides.get('raw_analog_cy', 0))
    return bytes(buf)


def _item_bytes(version: tuple, **overrides) -> bytes:
    sizes = {
        (3, 0, 0): 0x26,
        (3, 2, 0): 0x2A,
        (3, 6, 0): 0x2B,
        (3, 16, 0): 0x2D,
    }
    size = max(s for v, s in sizes.items() if v <= version)
    buf = bytearray(size)
    buf[0] = 0x3B  # ITEM_UPDATE
    struct.pack_into('>H', buf, 0x05, overrides.get('type', 5))
    buf[0x07] = overrides.get('state', 0)
    struct.pack_into('>f', buf, 0x08, overrides.get('direction', 1.0))
    struct.pack_into('>f', buf, 0x0C, overrides.get('vel_x', 0.0))
    struct.pack_into('>f', buf, 0x10, overrides.get('vel_y', 0.0))
    struct.pack_into('>f', buf, 0x14, overrides.get('pos_x', 0.0))
    struct.pack_into('>f', buf, 0x18, overrides.get('pos_y', 0.0))
    struct.pack_into('>H', buf, 0x1C, overrides.get('damage', 0))
    struct.pack_into('>f', buf, 0x1E, overrides.get('timer', 0.0))
    struct.pack_into('>I', buf, 0x22, overrides.get('id', 0))
    if version >= (3, 2, 0):
        struct.pack_into('>BBBB', buf, 0x26, *overrides.get('misc', (0, 0, 0, 0)))
    if version >= (3, 6, 0):
        struct.pack_into('>b', buf, 0x2A, overrides.get('owner', -1))
    return bytes(buf)


# ─────────────────────────────────────────────────────────────────────
# Byte-reader unit tests — exercise version cliffs and Optional handling.
# ─────────────────────────────────────────────────────────────────────

class CanonicalReaders(unittest.TestCase):

    def test_finite_int_or_zero_handles_missing_and_nonfinite_values(self):
        self.assertEqual(_finite_int_or_zero(None), 0)
        self.assertEqual(_finite_int_or_zero(float('nan')), 0)
        self.assertEqual(_finite_int_or_zero(float('inf')), 0)
        self.assertEqual(_finite_int_or_zero(float('-inf')), 0)
        self.assertEqual(_finite_int_or_zero(7.9), 7)

    def test_nonfinite_misc_action_state_survives_canonical_projection(self):
        console = object.__new__(melee.Console)
        console.slp_version_tuple = (3, 8, 0)
        console._current_stage = melee.Stage.FINAL_DESTINATION
        console._is_teams = False
        console._costumes = [0, 0, 0, 0]
        console._cpu_level = [0, 0, 0, 0]
        console._team_id = [0, 0, 0, 0]
        console._prev_gamestate = GameState()
        console._use_manual_bookends = False

        state = GameState()
        state.frame = 0
        event = bytearray(_post_bytes((3, 8, 0), misc_as=float('nan')))
        struct.pack_into('>i', event, 0x1, state.frame)
        console._Console__post_frame(state, bytes(event))

        self.assertTrue(math.isnan(state._canonical.ports[1].leader.post.misc_as))
        self.assertEqual(state.players[1].hitstun_frames_left, 0)

    def test_pre_required_fields(self):
        pre = _canonical.read_pre(
            _pre_bytes((0, 1, 0), random_seed=42, pos_x=1.5, joy_x=0.25), (0, 1, 0))
        self.assertEqual(pre.random_seed, 42)
        self.assertAlmostEqual(pre.position.x, 1.5)
        self.assertAlmostEqual(pre.joystick.x, 0.25)
        # All Optionals stay None on slp 0.1.
        self.assertIsNone(pre.raw_analog_x)
        self.assertIsNone(pre.percent)
        self.assertIsNone(pre.raw_analog_y)
        self.assertIsNone(pre.raw_analog_cstick_x)

    def test_pre_raw_analog_x_optional_on_old_slp(self):
        # slp 1.1 lacks raw_analog_x — must be None, NOT 0.
        pre = _canonical.read_pre(_pre_bytes((0, 2, 0)), (0, 2, 0))
        self.assertIsNone(pre.raw_analog_x)

    def test_pre_raw_analog_y_optional_on_old_slp(self):
        pre = _canonical.read_pre(_pre_bytes((1, 4, 0)), (1, 4, 0))
        # raw_analog_x present (slp 1.2+), raw_analog_y not (3.15+).
        self.assertIsNotNone(pre.raw_analog_x)
        self.assertIsNone(pre.raw_analog_y)

    def test_pre_modern(self):
        pre = _canonical.read_pre(
            _pre_bytes((3, 17, 0), raw_analog_x=-128, raw_analog_y=127, raw_analog_cx=42),
            (3, 17, 0))
        self.assertEqual(pre.raw_analog_x, -128)
        self.assertEqual(pre.raw_analog_y, 127)
        self.assertEqual(pre.raw_analog_cstick_x, 42)

    def test_post_required_fields(self):
        post = _canonical.read_post(
            _post_bytes((0, 1, 0), character=20, percent=37.5), (0, 1, 0))
        self.assertEqual(post.character, 20)
        self.assertAlmostEqual(post.percent, 37.5)
        # All Optionals stay None on slp 0.1.
        self.assertIsNone(post.state_age)
        self.assertIsNone(post.airborne)
        self.assertIsNone(post.velocities)
        self.assertIsNone(post.hurtbox_state)
        self.assertIsNone(post.animation_index)

    def test_post_airborne_is_int_not_bool(self):
        # Regression: airborne must be raw u8 (int 0/1), not bool, to match peppi.
        for raw in (0, 1):
            post = _canonical.read_post(
                _post_bytes((2, 0, 0), airborne=raw), (2, 0, 0))
            self.assertIsInstance(post.airborne, int)
            self.assertNotIsInstance(post.airborne, bool)
            self.assertEqual(post.airborne, raw)

    def test_post_velocities_optional_on_old_slp(self):
        post = _canonical.read_post(_post_bytes((3, 0, 0)), (3, 0, 0))
        self.assertIsNone(post.velocities)

    def test_post_velocities_present_on_modern(self):
        v = (1.0, -2.0, 3.0, -4.0, 5.0)
        post = _canonical.read_post(
            _post_bytes((3, 5, 0), velocities=v), (3, 5, 0))
        self.assertIsNotNone(post.velocities)
        self.assertAlmostEqual(post.velocities.self_x_air, 1.0)
        self.assertAlmostEqual(post.velocities.self_y, -2.0)
        self.assertAlmostEqual(post.velocities.self_x_ground, 5.0)

    def test_item_owner_optional_on_old_slp(self):
        # slp 3.0 has no owner field — peppi yields None.
        item = _canonical.read_item(_item_bytes((3, 0, 0)), (3, 0, 0))
        self.assertIsNone(item.owner)
        self.assertIsNone(item.misc)

    def test_item_owner_modern(self):
        item = _canonical.read_item(
            _item_bytes((3, 6, 0), owner=2, misc=(1, 2, 3, 4)), (3, 6, 0))
        self.assertEqual(item.owner, 2)
        self.assertEqual(item.misc, (1, 2, 3, 4))

    def test_item_owner_unowned_sentinel_preserved(self):
        # On slp 3.6+, -1 is the in-protocol "unowned" sentinel; libmelee must
        # surface it raw, not coerce to None.
        item = _canonical.read_item(_item_bytes((3, 6, 0), owner=-1), (3, 6, 0))
        self.assertEqual(item.owner, -1)


# ─────────────────────────────────────────────────────────────────────
# to_canonical_dict() shape tests: empty stage-event lists must serialize
# as [] not None (peppi semantics: None means "not in this slp version",
# [] means "active but no event this frame").
# ─────────────────────────────────────────────────────────────────────

class ToCanonicalDict(unittest.TestCase):
    def test_empty_stage_event_lists_serialize_as_list(self):
        # Bug: `if c.fod_platforms` truthiness check converted [] -> None.
        gs = GameState()
        gs._canonical = CanonicalFrame(
            id=42,
            ports={1: PortData(leader=Data(pre=Pre(), post=Post()))},
            items=[],
            fod_platforms=[],
            dreamland_whispys=[],
            stadium_transformations=[],
        )
        d = gs.to_canonical_dict()
        self.assertEqual(d['items'], [])
        self.assertEqual(d['fod_platforms'], [])
        self.assertEqual(d['dreamland_whispys'], [])
        self.assertEqual(d['stadium_transformations'], [])

    def test_inactive_stage_event_lists_serialize_as_none(self):
        gs = GameState()
        gs._canonical = CanonicalFrame(
            id=42,
            ports={1: PortData(leader=Data(pre=Pre(), post=Post()))},
            items=None,
            fod_platforms=None,
            dreamland_whispys=None,
            stadium_transformations=None,
        )
        d = gs.to_canonical_dict()
        self.assertIsNone(d['items'])
        self.assertIsNone(d['fod_platforms'])
        self.assertIsNone(d['dreamland_whispys'])
        self.assertIsNone(d['stadium_transformations'])

    def test_populated_fod_platforms_serialize_correctly(self):
        gs = GameState()
        gs._canonical = CanonicalFrame(
            id=42,
            ports={1: PortData(leader=Data(pre=Pre(), post=Post()))},
            fod_platforms=[FodPlatform(platform=1, height=12.5)],
        )
        d = gs.to_canonical_dict()
        self.assertEqual(d['fod_platforms'], [{'platform': 1, 'height': 12.5}])

    def test_follower_serializes_when_present(self):
        gs = GameState()
        gs._canonical = CanonicalFrame(
            ports={1: PortData(
                leader=Data(pre=Pre(), post=Post()),
                follower=Data(pre=Pre(), post=Post(character=11)),
            )},
        )
        d = gs.to_canonical_dict()
        self.assertIsNotNone(d['ports'][1]['follower'])
        self.assertEqual(d['ports'][1]['follower']['post']['character'], 11)


# ─────────────────────────────────────────────────────────────────────
# End-to-end tests against real fixtures.
# ─────────────────────────────────────────────────────────────────────

class EndToEnd(unittest.TestCase):
    def _replay(self, path: str, **kwargs) -> dict[int, dict]:
        """Drive libmelee in replay mode; return {frame_id: canonical dict}."""
        console = melee.Console(is_dolphin=False, path=path, **kwargs)
        self.assertTrue(console.connect())
        out: dict[int, dict] = {}
        while True:
            gs = console.step()
            if gs is None:
                break
            out[int(gs.frame)] = gs.to_canonical_dict()
        return out

    def test_test_game_1_canonical_shape(self):
        """test_game_1.slp is slp 3.6.1; canonical dict must have the expected
        Optional fields populated/None per the version cliffs."""
        canon = self._replay(
            'test_artifacts/test_game_1.slp', allow_old_version=False)
        self.assertEqual(len(canon), 1038)

        # Frame -123 (game start neutral). Both ports should be present.
        f = canon[-123]
        self.assertIn(1, f['ports'])
        self.assertIn(2, f['ports'])
        post1 = f['ports'][1]['leader']['post']
        # slp 3.6.1: velocities (3.5+) present, hitlag_left (3.8+) None,
        # animation_index (3.11+) None, instance_id (3.16+) None.
        self.assertIsNotNone(post1['velocities'])
        self.assertIsNone(post1['hitlag_left'])
        self.assertIsNone(post1['animation_index'])
        self.assertIsNone(post1['instance_id'])
        # airborne must be int 0 or 1, not bool.
        self.assertIsInstance(post1['airborne'], int)
        self.assertNotIsInstance(post1['airborne'], bool)

    def test_canonical_id_set_at_frame_start(self):
        """`_canonical.id` must equal `gamestate.frame` for every frame —
        regression guard for the id-stale-until-POST_FRAME bug."""
        console = melee.Console(
            is_dolphin=False, path='test_artifacts/test_game_1.slp')
        self.assertTrue(console.connect())
        while True:
            gs = console.step()
            if gs is None:
                break
            self.assertEqual(gs._canonical.id, gs.frame)
            self.assertEqual(gs.to_canonical_dict()['id'], gs.frame)

    def test_test_game_2_old_slp_optionals(self):
        """test_game_2.slp is slp 2.0.1: hurtbox_state (2.1+), velocities
        (3.5+), hitlag_left (3.8+), animation_index (3.11+) must all be None.
        raw_analog_y / raw_analog_cstick_* (3.15+/3.17+) must be None."""
        canon = self._replay(
            'test_artifacts/test_game_2.slp', allow_old_version=True)
        # Pick any frame deep into the game where players are present.
        for fid, f in canon.items():
            if fid >= 0 and f['ports']:
                break
        port = next(iter(f['ports']))
        post = f['ports'][port]['leader']['post']
        pre = f['ports'][port]['leader']['pre']
        self.assertIsNone(post['hurtbox_state'])
        self.assertIsNone(post['velocities'])
        self.assertIsNone(post['hitlag_left'])
        self.assertIsNone(post['animation_index'])
        self.assertIsNone(post['instance_id'])
        self.assertIsNone(pre['raw_analog_y'])
        self.assertIsNone(pre['raw_analog_cstick_x'])
        self.assertIsNone(pre['raw_analog_cstick_y'])

    def _assert_ic_follower_coverage(self, path: str):
        canon = self._replay(path, allow_old_version=True)
        ic_port = None
        with_follower = 0
        without_follower = 0
        for f in canon.values():
            for port, pd in f['ports'].items():
                if pd['follower'] is not None:
                    ic_port = port
                    self.assertEqual(pd['leader']['post']['character'], 10)
                    self.assertEqual(pd['follower']['post']['character'], 11)
        self.assertIsNotNone(ic_port, f"no follower in {path}")
        for f in canon.values():
            pd = f['ports'].get(ic_port)
            if pd is None:
                continue
            if pd['follower'] is not None:
                with_follower += 1
            else:
                without_follower += 1
        self.assertGreater(with_follower, 50, f"{path} must exercise alive-Nana")
        self.assertGreater(without_follower, 50, f"{path} must exercise dead-Nana")

    def test_ice_climbers_fod_follower_coverage(self):
        """IC + Fountain of Dreams clip across a Nana death."""
        self._assert_ic_follower_coverage('test_artifacts/fixture_ic_fod.slp')

    def test_ice_climbers_dl_follower_coverage(self):
        """IC + Dreamland clip across a Nana death."""
        self._assert_ic_follower_coverage('test_artifacts/fixture_ic_dl.slp')

    def test_rollback_resolution_changes_output(self):
        """test_game_1.slp is online-mode and contains rollbacks (per the peppi
        baseline harness comment). With `'first'` vs `'last'` we should yield
        different dicts on at least one frame."""
        first = self._replay(
            'test_artifacts/test_game_1.slp', rollback_resolution='first')
        last = self._replay(
            'test_artifacts/test_game_1.slp', rollback_resolution='last')
        self.assertEqual(set(first), set(last))
        diffs = sum(1 for fid in first if first[fid] != last[fid])
        self.assertGreater(diffs, 0,
                           "expected at least one frame to differ between "
                           "rollback_resolution='first' and 'last'")


# ─────────────────────────────────────────────────────────────────────
# peppi-py parity (skipped if peppi_py not installed). Exercises the
# strict dict-equality contract directly.
# ─────────────────────────────────────────────────────────────────────

class PeppiParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from peppi_py import read_frame_dicts  # noqa: F401
            from peppi_py.testing import dict_diff  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("peppi_py not installed")

    def _peppi_frames(self, path: str) -> dict[int, dict]:
        from peppi_py import read_frame_dicts
        return {int(f['id']): f for f in read_frame_dicts(path)}

    def _libmelee_frames(self, path: str, **kwargs) -> dict[int, dict]:
        # rollback_resolution='last' matches peppi's "final canonical state"
        # semantics for rollback-affected frames.
        console = melee.Console(
            is_dolphin=False, path=path,
            rollback_resolution='last', **kwargs)
        self.assertTrue(console.connect())
        out: dict[int, dict] = {}
        while True:
            gs = console.step()
            if gs is None:
                break
            out[int(gs.frame)] = gs.to_canonical_dict()
        return out

    def _assert_frame_parity(self, lm_frame: dict, peppi_frame: dict, fid: int, label: str):
        if lm_frame != peppi_frame:
            from peppi_py.testing import dict_diff
            self.fail(
                f'libmelee diverged from peppi (canonical) at frame {fid} of {label}:\n'
                + '\n'.join(dict_diff(lm_frame, peppi_frame))
            )

    def test_test_game_1_dict_parity(self):
        peppi = self._peppi_frames('test_artifacts/test_game_1.slp')
        lm = self._libmelee_frames('test_artifacts/test_game_1.slp')
        common = set(peppi) & set(lm)
        self.assertGreater(len(common), 100)
        # Spot-check a handful of frames; full equality is covered by the
        # peppi-py harness. Here we just want to catch schema regressions.
        for fid in sorted(common)[:20]:
            self._assert_frame_parity(lm[fid], peppi[fid], fid, 'test_game_1.slp')

    def _assert_full_parity(self, path: str):
        peppi = self._peppi_frames(path)
        lm = self._libmelee_frames(path, allow_old_version=True)
        common = sorted(set(peppi) & set(lm))
        self.assertEqual(len(common), len(peppi))
        for fid in common:
            self._assert_frame_parity(lm[fid], peppi[fid], fid, path)

    def test_ic_fod_dict_parity(self):
        self._assert_full_parity('test_artifacts/fixture_ic_fod.slp')

    def test_ic_dl_dict_parity(self):
        self._assert_full_parity('test_artifacts/fixture_ic_dl.slp')


class CrossShape(unittest.TestCase):
    """Structural assertion that libmelee `_canonical.py` dataclass field names
    match peppi-py's `frame.py` dataclass field names per type.

    Catches schema drift before it becomes a per-frame divergence at runtime.
    Skipped without peppi-py.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from peppi_py import frame  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("peppi_py not installed")

    def _names(self, cls) -> set[str]:
        import dataclasses as dc
        return {f.name for f in dc.fields(cls)}

    def test_pre_field_names_match(self):
        from peppi_py import frame as pp
        from melee import gamestate as lm
        self.assertEqual(self._names(lm.Pre), self._names(pp.Pre))

    def test_post_field_names_match(self):
        from peppi_py import frame as pp
        from melee import gamestate as lm
        self.assertEqual(self._names(lm.Post), self._names(pp.Post))

    def test_item_field_names_match(self):
        from peppi_py import frame as pp
        from melee import gamestate as lm
        self.assertEqual(self._names(lm.Item), self._names(pp.Item))

    def test_fod_platform_field_names_match(self):
        from peppi_py import frame as pp
        from melee import gamestate as lm
        self.assertEqual(self._names(lm.FodPlatform), self._names(pp.FodPlatform))

    def test_dreamland_whispy_field_names_match(self):
        from peppi_py import frame as pp
        from melee import gamestate as lm
        self.assertEqual(self._names(lm.DreamlandWhispy), self._names(pp.DreamlandWhispy))

    def test_stadium_transformation_field_names_match(self):
        from peppi_py import frame as pp
        from melee import gamestate as lm
        self.assertEqual(
            self._names(lm.CanonicalStadiumTransformation),
            self._names(pp.StadiumTransformation),
        )

    def test_frame_start_field_names_match(self):
        from peppi_py import frame as pp
        from melee import gamestate as lm
        self.assertEqual(self._names(lm.FrameStart), self._names(pp.FrameStart))

    def test_frame_end_field_names_match(self):
        from peppi_py import frame as pp
        from melee import gamestate as lm
        self.assertEqual(self._names(lm.FrameEnd), self._names(pp.FrameEnd))


if __name__ == '__main__':
    unittest.main()
