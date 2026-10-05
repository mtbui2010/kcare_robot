"""SwitchBot Bot (BLE button pusher on a light switch): ``turn_light``, ``light_state``.

The robot's container has no D-Bus, so BlueZ (bleak) is out; bluepy talks to
the adapter over raw HCI instead. Its helper needs network capabilities, set
once in the container::

    setcap cap_net_raw,cap_net_admin+eip ~/.local/lib/python3.10/site-packages/bluepy/bluepy-helper

and for light_state, which scans with its own HCI commands instead of the
kernel's mgmt discovery (that kept getting stuck here; see tools/hci_adv_sniff.c)::

    gcc -O2 -o ~/.local/bin/hci_adv_sniff tools/hci_adv_sniff.c
    sudo setcap cap_net_raw+eip ~/.local/bin/hci_adv_sniff

Devices are connections of type 'switchbot' (Connections panel, saved in the
site's connections.json), one per light: config {mac, loc, default}, e.g.

    switchbot_laundry_room  {'mac': 'EB:6B:01:06:62:34', 'loc': 'laundry room', 'default': true}
    switchbot_warehouse     {'mac': 'EC:0F:01:06:38:04', 'loc': 'warehouse'}

`loc` picks the light (its loc, connection id or MAC, or 'all'); without it,
the connection marked default.

A Bot in switch mode turns on / off; in press mode every action presses once.

Usage:
    turn_light::on                              # the default light (laundry room)
    turn_light::inputs='off', loc='warehouse'
    turn_light::inputs='on', loc='all'
    light_state                                 # on / off, battery, mode of every light
    light_state::loc='warehouse'
"""
import json
import os
import subprocess
import threading
import time

from robot_agent.skills import log_data
from robot_agent.utils import exception_handler

_TX = 'cba20002-224d-11e6-9fb8-0002a5d5c51b'          # write commands here
_RX = 'cba20003-224d-11e6-9fb8-0002a5d5c51b'          # the result comes back as a notification
_COMMANDS = {'press': b'\x57\x01\x00', 'on': b'\x57\x01\x01', 'off': b'\x57\x01\x02'}
_RESULTS = {1: 'ok', 3: 'low battery', 5: 'busy / password protected', 11: 'action failed'}
# Spoken / typed forms of the actions.
_ALIASES = {'on': 'on', 'off': 'off', 'press': 'press', 'toggle': 'press',
            '켜': 'on', '켜줘': 'on', '꺼': 'off', '꺼줘': 'off', 'bật': 'on', 'tắt': 'off',
            'true': 'on', 'false': 'off', '1': 'on', '0': 'off'}
_DEFAULTS = {'scan_sec': 5.0, 'retries': 3}

_lock = threading.Lock()           # one BLE operation at a time: there is one adapter

# Last known state per MAC — from an advertisement or a confirmed on/off — for
# light_state while another program holds the adapter's scan: bluetoothd scans
# with duplicate filtering, so each device is reported once when that scan
# starts and nothing fresh can be heard afterwards.
_STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'switchbot_state.json')


def _load_states() -> dict:
    try:
        with open(_STATE_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _remember(mac: str, state: dict, source: str) -> None:
    states = _load_states()
    states[mac] = {**states.get(mac, {}), **{k: v for k, v in state.items() if k != 'rssi'},
                   'source': source, 'ts': time.time()}
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        with open(_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(states, f, indent=1)
    except OSError:
        pass


def _config(kwargs: dict) -> dict:
    cfg = dict(_DEFAULTS)
    for k in ('scan_sec', 'retries'):
        if k in kwargs:
            cfg[k] = kwargs.pop(k)
    return cfg


def _devices() -> list:
    """[{id, mac, loc, default}] — the 'switchbot' connections of the robot."""
    from robot_agent.state import current
    return current().dm.switchbots()


def _norm(s) -> str:
    return ' '.join(str(s).strip().lower().replace('_', ' ').split())


def _targets(loc) -> list:
    """[(loc, MAC)] for `loc`: a light's loc, its connection id or MAC, or
    'all'; None → the connection marked default (else the first)."""
    devices = _devices()
    if not devices:
        raise ValueError("no SwitchBot configured: add a 'switchbot' connection in the Connections panel")
    if loc is None or str(loc).strip() == '':
        d = next((d for d in devices if d['default']), devices[0])
        return [(d['loc'], d['mac'])]
    want = _norm(loc)
    if want == 'all':
        return [(d['loc'], d['mac']) for d in devices]
    for d in devices:
        if want in (_norm(d['loc']), _norm(d['id']), d['mac'].lower()):
            return [(d['loc'], d['mac'])]
    raise ValueError(f'no SwitchBot at "{loc}" — configured: '
                     + ', '.join(d['loc'] for d in devices) + ' or all')


def _bot_state(service_data: bytes) -> dict:
    """State of a Bot from its 0xfd3d service data (after the UUID)."""
    data = service_data
    if len(data) >= 3 and data[0] & 0x7f == 0x48:                # 'H' = Bot (WoHand)
        switch = bool(data[1] & 0x80)
        return {'mode': 'switch' if switch else 'press',
                'is_on': (not bool(data[1] & 0x40)) if switch else None,
                'battery': data[2] & 0x7f}
    return {}


def _ad_service_data(ad: bytes) -> bytes:
    """The 0xfd3d service data in raw advertising data, or b''."""
    i = 0
    while i < len(ad):
        n = ad[i]
        if n == 0 or i + 1 + n > len(ad):
            break
        if ad[i + 1] == 0x16 and ad[i + 2:i + 4] == b'\x3d\xfd':  # service data, UUID 0xfd3d (LE)
            return ad[i + 4:i + 1 + n]
        i += 1 + n
    return b''


class _AdapterBusy(RuntimeError):
    """Another program holds discovery on the adapter."""


def _scan_bluepy(wanted: set, seconds: float) -> dict:
    """{MAC: state} from a bluepy scan. Raises _AdapterBusy when another
    program is already scanning.

    Not Scanner.start(): on 'busy' it sends 'scanend' to stop the other
    program's scan, which the kernel refuses (code 13) — and after that the
    kernel's discovery state stayed stuck (every later scan 'busy', stop
    'rejected') until the adapter was power-cycled. Here a busy adapter is
    left alone. Used only when tools/hci_adv_sniff is not installed."""
    from bluepy.btle import BTLEManagementError, Scanner
    sc = Scanner(0)
    sc.clear()
    sc._startHelper(iface=0)
    try:
        sc._mgmtCmd('le on')
        sc._writeCmd('scan\n')
        rsp = sc._waitResp('mgmt')
        if rsp['code'][0] != 'success':
            raise _AdapterBusy(rsp['code'][0])
        sc.process(seconds)
        try:
            sc.stop()
        except BTLEManagementError:
            pass
    finally:
        try:
            sc._stopHelper()
        except Exception:
            pass
    out = {}
    for d in sc.getDevices():
        if d.addr.upper() in wanted:
            st = {'rssi': d.rssi}
            for adtype, _desc, value in d.getScanData():
                if adtype == 0x16 and value[:4].lower() == '3dfd':
                    st.update(_bot_state(bytes.fromhex(value[4:])))
            out[d.addr.upper()] = st
    return out


_SNIFFER = os.path.expanduser('~/.local/bin/hci_adv_sniff')     # tools/hci_adv_sniff.c, with CAP_NET_RAW


def _hci_scan(wanted: set, seconds: float):
    """({MAC: state}, mode) from ``hci_adv_sniff --scan``, or None when it is
    not installed (build: see tools/hci_adv_sniff.c).

    It drives the controller with HCI scan commands itself, so it does not
    depend on the kernel's mgmt discovery — which on this robot kept getting
    stuck ('busy' / 'rejected' until the adapter was power-cycled). mode is
    'own-extended' / 'own-legacy' (it scanned), 'shared' (another program
    was scanning; left alone and listened to) or 'failed …'."""
    if not os.access(_SNIFFER, os.X_OK):
        return None
    run = subprocess.run([_SNIFFER, str(seconds), '--scan'], capture_output=True, text=True,
                         timeout=seconds + 10)
    mode = next((l.split(':', 1)[1].strip() for l in run.stderr.splitlines() if l.startswith('scan:')), '?')
    out = {}
    for line in run.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in wanted:
            sd = _ad_service_data(bytes.fromhex(parts[2]))
            if sd:
                out[parts[0]] = {'rssi': int(parts[1]), **_bot_state(sd)}
    return out, mode


def _scan(macs, seconds: float) -> dict:
    """{MAC: state} for the wanted devices heard within `seconds`.

    Raises RuntimeError when nothing can be heard because another program
    holds the scan (light_state then gives the last known state)."""
    wanted = {m.upper() for m in macs}
    hit = _hci_scan(wanted, seconds)
    if hit is not None:
        seen, mode = hit
        if seen:
            return seen
        if mode == 'shared':
            # bluetoothd scans with duplicate filtering: each device is
            # reported once, when that scan starts, and not again.
            raise RuntimeError('another program is scanning Bluetooth (e.g. Settings → Bluetooth open '
                               'on the robot desktop); nothing fresh can be heard until it stops')
        if mode.startswith('failed'):
            raise RuntimeError(f'Bluetooth scan failed ({mode})')
        return {}                                   # scanned, not heard: out of range / battery
    # Sniffer not installed: a bluepy scan through the kernel's mgmt discovery.
    try:
        return _scan_bluepy(wanted, seconds)
    except _AdapterBusy as e:
        raise RuntimeError(f'the Bluetooth adapter is busy ({e}); install tools/hci_adv_sniff '
                           'so light_state can scan by itself') from None


def _send(mac: str, action: str) -> dict:
    """Connect straight to `mac` (a Bot uses a random static address, so no
    scan is needed), write the command, wait for the result notification."""
    from bluepy.btle import ADDR_TYPE_RANDOM, DefaultDelegate, Peripheral

    class _Reply(DefaultDelegate):
        data = None

        def handleNotification(self, _handle, data):
            self.data = bytes(data)

    reply = _Reply()
    p = Peripheral(mac, ADDR_TYPE_RANDOM, iface=0)
    try:
        p.setDelegate(reply)
        rx = p.getCharacteristics(uuid=_RX)[0]
        p.writeCharacteristic(rx.getHandle() + 1, b'\x01\x00', withResponse=True)   # enable notifications
        p.getCharacteristics(uuid=_TX)[0].write(_COMMANDS[action], withResponse=True)
        deadline = time.time() + 5.0
        while reply.data is None and time.time() < deadline:
            p.waitForNotifications(0.5)
    finally:
        try:
            p.disconnect()
        except Exception:
            pass
    code = reply.data[0] if reply.data else -1
    return {'ok': code == 1, 'code': code, 'result': _RESULTS.get(code, 'no reply' if code < 0 else f'code {code}')}


@exception_handler
def turn_light(node=None, **kwargs):
    """Turn a light on / off with its SwitchBot Bot.

    Params:
        inputs: 'on' | 'off' | 'press' (also 켜 / 꺼, bật / tắt).
        loc: the light — a switchbot connection's loc ('laundry room'), its
            id or MAC, or 'all'; default the connection marked default.
            (`device` is accepted as an old name for it.)
        retries, scan_sec: override the defaults (3, 5 s).

    Returns ``{'isdone', 'results': {loc: {ok, result, attempts, ...}}}`` —
    isdone only when every addressed device confirmed.
    """
    cfg = _config(kwargs)
    action = _ALIASES.get(str(kwargs.pop('inputs', '') or '').strip().lower())
    if action is None:
        raise ValueError("inputs must be 'on', 'off' or 'press'")
    loc = kwargs.pop('loc', None)
    targets = _targets(loc if loc is not None else kwargs.pop('device', None))

    results = {}
    with _lock:
        for name, mac in targets:
            last = ''
            for attempt in range(1, int(cfg['retries']) + 1):
                try:
                    res = _send(mac, action)
                    if res['ok'] or res['code'] > 0:       # the Bot answered: do not repeat the motion
                        results[name] = {**res, 'mac': mac, 'attempts': attempt}
                        break
                    last = res['result']
                except Exception as e:                     # out of range / busy / GATT: often transient
                    last = f'{type(e).__name__}: {e}'
                time.sleep(1.0)
            else:
                results[name] = {'ok': False, 'mac': mac, 'result': last, 'attempts': int(cfg['retries'])}
            if results[name]['ok'] and action in ('on', 'off'):
                _remember(mac, {'is_on': action == 'on'}, 'command')
            log_data({'msg': f"turn_light {action} {name}: {results[name]['result']}"})
    return {'isdone': all(r['ok'] for r in results.values()), 'action': action, 'results': results}


@exception_handler
def light_state(node=None, **kwargs):
    """On / off, battery and mode of SwitchBot Bots, read from their
    advertisements (no connection). ``loc`` as for turn_light; default all.

    A device not heard now — the adapter busy with another program's scan, or
    out of range — is reported from its last known state, marked
    ``stale: True`` with ``age_sec`` and ``source`` ('advert' or 'command'),
    never passed off as fresh. isdone is False only when a device has no state
    at all."""
    cfg = _config(kwargs)
    targets = _targets(kwargs.pop('loc', None) or kwargs.pop('device', None) or 'all')
    note = ''
    with _lock:
        try:
            seen = _scan([m for _, m in targets], float(cfg['scan_sec']))
        except RuntimeError as e:                  # busy adapter, nothing heard
            seen, note = {}, str(e)
    for mac, st in seen.items():
        _remember(mac, st, 'advert')
    known = _load_states()
    states = {}
    for name, mac in targets:
        if mac in seen:
            states[name] = {'mac': mac, **seen[mac], 'stale': False}
        elif mac in known:
            k = known[mac]
            states[name] = {'mac': mac, **{x: k[x] for x in ('is_on', 'mode', 'battery') if x in k},
                            'stale': True, 'source': k.get('source'), 'age_sec': round(time.time() - k['ts'])}
        else:
            states[name] = {'mac': mac, 'seen': False}
    ret = {'isdone': all('is_on' in st or 'mode' in st for st in states.values()), 'states': states}
    if note or any(st.get('stale') for st in states.values()):
        ret['msg'] = note or 'some devices not heard now; their last known state is given'
    return ret
