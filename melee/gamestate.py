""" Gamestate is a single snapshot in time of the game that represents all necessary information
        to make gameplay decisions
"""
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Optional

import numpy as np

import melee
from melee import enums

@dataclass(slots=True, unsafe_hash=True)
class Position:
    """Dataclass for position types. Has (x, y) coords."""
    x: np.float32 = np.float32(0)
    y: np.float32 = np.float32(0)

Speed = Position
Cursor = Position

@dataclass(slots=True)
class ECB:
    """ECBs (Environmental collision box) info. It's a diamond with four points that define it."""
    top: Position = field(default_factory=Position)
    bottom: Position = field(default_factory=Position)
    left: Position = field(default_factory=Position)
    right: Position = field(default_factory=Position)

@dataclass(slots=True)
class FoDPlatforms:
    """Dataclass for the platforms on the Fountain of Dreams stage."""
    # Initial values are the same each time, found by experimentation and
    # extrapolating backwards after the platforms start moving.
    left: np.float32 = np.float32(20)
    right: np.float32 = np.float32(28)

class WhispyBlowDirection(Enum):
    NONE = np.uint8(0)
    LEFT = np.uint8(1)
    RIGHT = np.uint8(2)

# https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md#stadium-transformations
class StadiumTransformationEvent(Enum):
    """Pokemon Stadium transformation event types"""
    FINISHED = 0
    INITIALIZE = 2
    ON_MONITOR = 3
    PREVIOUS_RECEDING = 4
    NEW_RISING = 5
    FINALIZE = 6

class StadiumTransformationType(Enum):
    """Pokemon Stadium transformation types"""
    FIRE = 3
    GRASS = 4
    NORMAL = 5
    ROCK = 6
    WATER = 9

@dataclass(slots=True)
class StadiumTransformation:
    """Current Pokemon Stadium transformation state"""
    event: StadiumTransformationEvent = StadiumTransformationEvent.FINISHED
    type: StadiumTransformationType = StadiumTransformationType.NORMAL

# ────────────────────────────────────────────────────────────────────────
# Canonical schema (mirrors peppi-py's read_frame_dicts output verbatim).
# These dataclasses are the source of truth for GameState.to_canonical_dict().
# Field names, ordering, and types match peppi::frame::transpose so that
# online libmelee and offline peppi produce byte-for-byte identical dicts.
# ────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Velocity:
    x: float = 0.0
    y: float = 0.0

@dataclass(slots=True)
class Velocities:
    self_x_air: float = 0.0
    self_y: float = 0.0
    knockback_x: float = 0.0
    knockback_y: float = 0.0
    self_x_ground: float = 0.0

@dataclass(slots=True)
class TriggersPhysical:
    l: float = 0.0
    r: float = 0.0

@dataclass(slots=True)
class Pre:
    random_seed: int = 0
    state: int = 0                                    # raw u16
    position: Position = field(default_factory=Position)
    direction: float = 0.0                            # raw f32, NOT bool
    joystick: Position = field(default_factory=Position)   # range [-1, 1]
    cstick: Position = field(default_factory=Position)     # range [-1, 1]
    triggers: float = 0.0                             # combined [0, 1]
    buttons: int = 0                                  # u32 processed bitmask
    buttons_physical: int = 0                         # u16
    triggers_physical: TriggersPhysical = field(default_factory=TriggersPhysical)
    raw_analog_x: Optional[int] = None                # slp 1.2+, raw i8
    percent: Optional[float] = None                   # slp 1.4+
    raw_analog_y: Optional[int] = None                # slp 3.15+
    raw_analog_cstick_x: Optional[int] = None         # slp 3.17+
    raw_analog_cstick_y: Optional[int] = None         # slp 3.17+

@dataclass(slots=True)
class Post:
    character: int = 0
    # Action-state id (slp spec calls this "state"; we use "action" to match
    # libmelee's PlayerState.action and MDS column naming. state_age and
    # state_flags keep the historical "state_" prefix.
    action: int = 0
    position: Position = field(default_factory=Position)
    direction: float = 0.0                            # raw f32
    percent: float = 0.0
    shield: float = 60.0
    last_attack_landed: int = 0
    combo_count: int = 0
    last_hit_by: int = 0
    stock: int = 0                                    # slp spec: "stocks"
    state_age: Optional[float] = None                 # slp 0.2+, age of `action`
    state_flags: Optional[tuple] = None               # slp 2.0+, 5×u8 on `action`
    misc_as: Optional[float] = None                   # slp 2.0+
    airborne: Optional[int] = None                    # slp 2.0+, raw u8 (0=ground, 1=air)
    ground: Optional[int] = None                      # slp 2.0+
    jumps_used: Optional[int] = None                  # slp 2.0+, spec: "jumps"
    l_cancel: Optional[int] = None                    # slp 2.0+
    hurtbox_state: Optional[int] = None               # slp 2.1+, raw u8
    velocities: Optional[Velocities] = None           # slp 3.5+
    hitlag_left: Optional[float] = None               # slp 3.8+, raw f32, spec: "hitlag"
    animation_index: Optional[int] = None             # slp 3.11+
    last_hit_by_instance: Optional[int] = None        # slp 3.16+
    instance_id: Optional[int] = None                 # slp 3.16+
    # ECB will land here (Optional[CanonicalECB]) once the peppi fork ships
    # the right byte offsets; legacy playerstate.ecb_* in console.py is
    # the temporary stand-in until then.

@dataclass(slots=True)
class Data:
    pre: Pre = field(default_factory=Pre)
    post: Post = field(default_factory=Post)

@dataclass(slots=True)
class PortData:
    leader: Data = field(default_factory=Data)
    follower: Optional[Data] = None

@dataclass(slots=True)
class Item:
    type: int = 0
    state: int = 0
    direction: float = 0.0
    velocity: Velocity = field(default_factory=Velocity)
    position: Position = field(default_factory=Position)
    damage: int = 0
    timer: float = 0.0
    id: int = 0
    misc: Optional[tuple] = None                      # slp 3.2+, 4×u8
    owner: Optional[int] = None                       # slp 3.6+, raw i8 (-1 unowned, 0..3 port)
    instance_id: Optional[int] = None                 # slp 3.16+

@dataclass(slots=True)
class FrameStart:
    random_seed: int = 0
    scene_frame_counter: Optional[int] = None         # slp 3.10+

@dataclass(slots=True)
class FrameEnd:
    latest_finalized_frame: Optional[int] = None      # slp 3.7+

@dataclass(slots=True)
class FodPlatform:
    platform: int = 0
    height: float = 0.0

@dataclass(slots=True)
class DreamlandWhispy:
    direction: int = 0

@dataclass(slots=True)
class CanonicalStadiumTransformation:
    event: int = 0
    type: int = 0

@dataclass(slots=True)
class StartPlayer:
    port: int = 0
    character: int = 0
    type: int = 0
    stocks: int = 0
    costume: int = 0
    team: Optional[int] = None
    cpu_level: Optional[int] = None
    name_tag: str = ""
    display_name: str = ""
    connect_code: str = ""

@dataclass(slots=True)
class GameStart:
    stage: int = 0
    is_teams: bool = False
    slp_version: tuple = (0, 0, 0)
    players: dict = field(default_factory=dict)        # {port: StartPlayer}

@dataclass(slots=True)
class OnlineState:
    """libmelee-only state never present in the canonical per-frame dict."""
    menu_state: enums.Menu = enums.Menu.IN_GAME
    submenu: enums.SubMenu = enums.SubMenu.UNKNOWN_SUBMENU
    menu_selection: int = 0
    ready_to_start: bool = False
    is_frozen_ps: bool = False
    cursors: dict = field(default_factory=dict)               # {port: Position}
    coin_down: dict = field(default_factory=dict)             # {port: bool}
    controller_status: dict = field(default_factory=dict)     # {port: enums.ControllerStatus}
    character_selected: dict = field(default_factory=dict)    # {port: enums.Character}
    is_holding_cpu_slider: dict = field(default_factory=dict) # {port: bool}
    started_at: str = ""
    played_on: str = ""
    console_nick: str = ""

@dataclass(slots=True)
class CanonicalFrame:
    """The per-frame data that to_canonical_dict() serializes.

    Lives on GameState as `_canonical` while the legacy PlayerState tree is
    still around. Once consumers migrate, this becomes the only state.
    """
    id: int = 0
    start: Optional[FrameStart] = None
    end: Optional[FrameEnd] = None
    ports: dict = field(default_factory=dict)         # {port: PortData}
    # `items` stays None for slp <3.0 (the version that introduced item events)
    # and is initialized to [] at FRAME_START for slp 3.0+ so frames with no
    # items still emit an empty list — matching peppi's Option<Vec<Item>>.
    items: Optional[list] = None                      # [Item] | None
    fod_platforms: Optional[list] = None              # [FodPlatform]
    dreamland_whispys: Optional[list] = None          # [DreamlandWhispy]
    stadium_transformations: Optional[list] = None    # [CanonicalStadiumTransformation]


@dataclass(slots=True)
class GameState:
    """Represents the state of a running game of Melee at a given moment in time"""
    frame: int = -10000
    """int: The current frame number. Monotonically increases. Can be negative."""
    stage: enums.Stage = enums.Stage.FINAL_DESTINATION
    """enums.Stage: The current stage being played on"""

    # Stage-specific state
    whispy: Optional[WhispyBlowDirection] = None
    fod_platforms: Optional[FoDPlatforms] = None
    stadium_transformation: Optional[StadiumTransformation] = None

    menu_state: enums.Menu = enums.Menu.IN_GAME
    """enums.MenuState: The current menu scene, such as IN_GAME, or STAGE_SELECT"""
    submenu: enums.SubMenu = enums.SubMenu.UNKNOWN_SUBMENU
    """(enums.SubMenu): The current sub-menu"""
    players: dict[int, 'PlayerState'] = field(default_factory=dict)
    """(dict of int - gamestate.PlayerState): Dict of PlayerState objects. Key is controller port"""
    projectiles: list['Projectile'] = field(default_factory=list)
    """(list of Projectile): All projectiles (items) currently existing"""
    ready_to_start: bool = False
    """(bool): Is the 'ready to start' banner showing at the character select screen?"""
    is_teams: bool = False
    """(bool): Is this a teams game?"""
    distance: float = 0.0
    """(float): Euclidian distance between the two players. (or just Popo for climbers)"""
    menu_selection: int = 0
    """(int): The index of the selected menu item for when in menus."""
    startAt: str = ""
    """(string): Timestamp string of when the game started. Such as '2018-06-22T07:52:59Z'"""
    playedOn: str = ""
    """(string): Platform the game was played on (values include dolphin, console, and network). Might be blank."""
    consoleNick: str = ""
    """(string): The name of the console the replay was created on. Might be blank."""
    _newframe: bool = True
    custom: dict = field(default_factory=dict)
    """(dict): Custom fields to be added by the user"""

    # Canonical-schema state (peppi-py parity). Populated alongside the legacy
    # PlayerState tree during the migration; will become the only state once
    # downstream consumers move off the deprecated attributes.
    _canonical: CanonicalFrame = field(default_factory=CanonicalFrame)
    online: OnlineState = field(default_factory=OnlineState)
    game_start: Optional[GameStart] = None

    def to_canonical_dict(self) -> dict:
        """Return the per-frame dict matching peppi_py.read_frame_dicts.

        Equality with `peppi_py.read_frame_dicts(slp)[i]` is the contract that
        gates imitation-learning pipelines: training-from-replay (peppi) and
        acting-online (libmelee) must see byte-for-byte identical observations.
        See peppi-py/tests/test_libmelee_parity.py for the harness.
        """
        return _canon_to_dict(self._canonical)


def _canon_to_dict(obj):
    """Fast canonical-schema serializer; drop-in replacement for asdict.

    Skips asdict's deepcopy of every container — the canonical tree is
    primitives + dataclasses + list[dataclass] + dict[int, dataclass], so a
    shallow recursion suffices and the consumer doesn't mutate values.
    Matters at 60Hz: per-frame dict cost dominates online inference latency.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _canon_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_canon_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _canon_to_dict(v) for k, v in obj.items()}
    return obj

@dataclass(slots=True)
class UnknownAnimation:
    value: int = -1

@dataclass(slots=True)
class PlayerState:
    """ Represents the state of a single player """
    # This value is what the character currently is IN GAME
    #   So this will have no meaning while in menus
    #   Also, this will change dynamically if you change characters
    #       IE: Shiek/Zelda
    character: enums.Character = enums.Character.UNKNOWN_CHARACTER
    """(enum.Character): The player's current character"""
    # This value is what character is selected at the character select screen
    #   Don't use this value when in-game
    character_selected: enums.Character = enums.Character.UNKNOWN_CHARACTER
    position: Position = field(default_factory=Position)
    """(Position): x, y character position"""
    percent: float = 0
    """(int): The player's damage"""
    shield_strength: float = 60.
    """(float): The player's shield strength (max 60). Shield breaks at 0"""
    is_powershield: bool = False
    """(bool): Is the current action a Powershield? (not directly determinable via action states)"""
    stock: int = 0
    """(int): The player's remaining stock count"""
    facing: bool = True
    """(bool): Is the character facing right? (left is False). Characters in Melee must always be facing left or right"""
    action: enums.Action | UnknownAnimation = field(default_factory=UnknownAnimation)
    """(enum.Action): The current action (or animation) the character is in"""
    action_frame: int = 0
    """(int): What frame of the Action is the character in? Indexed from 1."""
    invulnerable: bool = False
    """(bool): Is the player invulnerable?"""
    hitlag_left: int = 0
    """(bool): How many more frames of hitlag there is"""
    hitstun_frames_left: int = 0
    """(int): How many more frames of hitstun there is"""
    jumps_left: int = 0
    """(int): Number of jumps available. Including ground jump. Will be 2 for most characters on ground."""
    on_ground: bool = True
    """(bool): Is the character on the ground?"""
    speed_air_x_self: float = 0
    """(float): Self-induced horizontal air speed"""
    speed_y_self: float = 0
    """(float): Self-induced vertical speed"""
    speed_x_attack: float = 0
    """(float): Attack-induced horizontal speed"""
    speed_y_attack: float = 0
    """(float): Attack-induced vertical speed"""
    speed_ground_x_self: float = 0
    """(float): Self-induced horizontal ground speed"""
    nana: Optional['PlayerState'] = None
    """(enums.PlayerState): Additional player state for Nana, if applicable.
            If the character is not Ice Climbers, Nana will be None.
            Will also be None if this player state is Nana itself.
            Lastly, the secondary climber is called 'Nana' here, regardless of the costume used."""
    cursor: Cursor = field(default_factory=Cursor)
    """(Position): x, y cursor position"""
    coin_down: bool = False
    """(bool): Is the player's character selection coin placed down? (Does not work in Slippi selection screen)"""
    controller_status: enums.ControllerStatus = enums.ControllerStatus.CONTROLLER_UNPLUGGED
    """(enums.ControllerStatus): Status of the player's controller."""
    off_stage: bool = False
    """(bool): Helper variable to say if the character is 'off stage'. """
    iasa: int = 0
    moonwalkwarning: bool = False
    """(bool): Helper variable to tell you that if you dash back right now, it'll moon walk"""
    controller_state: 'melee.ControllerState' = field(default_factory=lambda: melee.ControllerState())
    """(controller.ControllerState): What buttons were pressed for this character"""
    ecb: ECB = field(default_factory=ECB)
    ecb_right: tuple = (0, 0)
    """(float, float): Right edge of the ECB. (x, y) offset from player's center."""
    ecb_left: tuple = (0, 0)
    """(float, float): Left edge of the ECB. (x, y) offset from player's center."""
    ecb_top: tuple = (0, 0)
    """(float, float): Top edge of the ECB. (x, y) offset from player's center."""
    ecb_bottom: tuple = (0, 0)
    """(float, float): Bottom edge of the ECB. (x, y) offset from player's center."""
    costume: int = 0
    """(int): Index for which costume the player is wearing"""
    cpu_level: int = 0
    """(int): CPU level of player. 0 for a libmelee-controller bot or human player."""
    is_holding_cpu_slider: bool = False
    """(bool): Is the player holding the CPU slider in the character select screen?"""
    nickName: str = ""
    """(string): The in-game nickname for the player. Might be blank."""
    connectCode: str = ""
    """(string): The rollback connect code for the player. Might be blank."""
    displayName: str = ""
    """(string): The Slippi Online display name for the play. Might be blank"""
    team_id: int = 0
    """(int): The team ID of the player. This is different than costume, and only relevant during teams."""

@dataclass(slots=True)
class UnknownProjectileType:
    """ Represents an unknown projectile type """
    value: np.uint16

@dataclass(slots=True, unsafe_hash=True)
class Projectile:
    """ Represents the state of a projectile (items, lasers, etc...) """
    # frame: int = 0
    position: Position = field(default_factory=Position)
    """(Position): x, y projectile position"""
    speed: Speed = field(default_factory=Speed)
    """(Position): x, y projectile speed"""
    owner: int = -1
    """(int): Player port of the projectile's owner. -1 for no owner"""
    type: enums.ProjectileType | UnknownProjectileType = enums.ProjectileType.UNKNOWN_PROJECTILE
    """(enums.ProjectileType): Which actual projectile type this is"""
    expiration_frames: float = 0
    """(int): How long the item has been out"""
    subtype: int = 0
    """(int): The subtype of the item. Many projectiles have 'subtypes' that make them different. They're all different, so it's not an enum"""
    spawn_id: np.uint32 = np.uint32(0)

def port_detector(gamestate, character, costume):
    """Autodiscover what port the given character is on

    Slippi Online assigns us a random port when playing online. Find out which we are

    Returns:
        [1-4]: The given character belongs to the returned port
        0: We don't know.

    Args:
        gamestate: Current gamestate
        character: The character we know we picked
        costume: Costume index we picked
    """
    detected_port = 0
    for i, player in gamestate.players.items():
        if player.character == character and player.costume == costume:
            if detected_port > 0:
                return 0
            detected_port = i

    return detected_port
