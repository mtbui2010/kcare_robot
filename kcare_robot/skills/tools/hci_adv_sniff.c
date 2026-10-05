/*
 * hci_adv_sniff — scan for / listen to BLE advertising reports on an HCI adapter.
 *
 * Used by kcare_robot/skills/switchbot.py (light_state). Scanning through the
 * kernel's mgmt discovery (bluepy) kept getting stuck on this robot — every
 * start 'busy', every stop 'rejected' until the adapter was power-cycled — so
 * with --scan this program drives the controller itself with HCI commands:
 *
 *   LE Set Extended Scan Parameters / Enable   (BT 5; what the kernel uses)
 *   LE Set Scan Parameters / Enable            (fallback: older controllers)
 *
 * active scanning (scan responses too), duplicate filtering OFF (every report),
 * and turns scanning off again only if it was the one that turned it on. When
 * the controller is already scanning for someone else (Command Disallowed),
 * that scan is left alone and its reports are read.
 *
 * Without --scan it only listens (the reports of a scan someone else runs).
 * Raw HCI needs CAP_NET_RAW, hence a small program with that capability rather
 * than granting it to python.
 *
 * Build and grant (in the robot container):
 *   gcc -O2 -o ~/.local/bin/hci_adv_sniff hci_adv_sniff.c
 *   sudo setcap cap_net_raw+eip ~/.local/bin/hci_adv_sniff
 *
 * Usage: hci_adv_sniff [seconds=5] [hci index=0] [--scan]
 * stdout, one line per report:  AA:BB:CC:DD:EE:FF <rssi> <advertising data hex>
 * stderr, one line:             scan: own-extended | own-legacy | shared | listen-only | failed <why>
 */
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define BTPROTO_HCI 1
#define SOL_HCI 0
#define HCI_FILTER 2
#define HCI_CHANNEL_RAW 0
#define HCI_COMMAND_PKT 0x01
#define HCI_EVENT_PKT 0x04
#define EVT_CMD_COMPLETE 0x0e
#define EVT_CMD_STATUS 0x0f
#define EVT_LE_META 0x3e

#define OP(ocf) ((uint16_t)((0x08 << 10) | (ocf)))      /* OGF 0x08: LE controller */
#define LE_SET_SCAN_PARAMS      OP(0x000b)
#define LE_SET_SCAN_ENABLE      OP(0x000c)
#define LE_SET_EXT_SCAN_PARAMS  OP(0x0041)
#define LE_SET_EXT_SCAN_ENABLE  OP(0x0042)

#define ST_UNKNOWN_CMD 0x01
#define ST_DISALLOWED  0x0c

struct sockaddr_hci { sa_family_t hci_family; unsigned short hci_dev; unsigned short hci_channel; };
struct hci_filter { uint32_t type_mask; uint32_t event_mask[2]; uint16_t opcode; };

static int s = -1;
/* SIGTERM / SIGINT end the read loop early but still turn our scan off. */
static volatile sig_atomic_t stop_now = 0;
static void on_signal(int sig) { (void)sig; stop_now = 1; }

static double now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

static void print_report(const uint8_t *addr, int rssi, const uint8_t *ad, int n) {
    printf("%02X:%02X:%02X:%02X:%02X:%02X %d ", addr[5], addr[4], addr[3], addr[2], addr[1], addr[0], rssi);
    for (int i = 0; i < n; i++) printf("%02x", ad[i]);
    printf("\n");
}

static void handle_le_meta(const uint8_t *buf, int len) {
    int sub = buf[3], num = buf[4], p = 5;
    for (int k = 0; k < num; k++) {
        if (sub == 0x02) {                       /* legacy advertising report */
            if (p + 9 > len) return;
            int n = buf[p + 8];
            if (p + 10 + n > len) return;
            print_report(&buf[p + 2], (int8_t)buf[p + 9 + n], &buf[p + 9], n);
            p += 10 + n;
        } else if (sub == 0x0d) {                /* extended advertising report */
            if (p + 24 > len) return;
            int n = buf[p + 23];
            if (p + 24 + n > len) return;
            print_report(&buf[p + 3], (int8_t)buf[p + 13], &buf[p + 24], n);
            p += 24 + n;
        } else {
            return;
        }
    }
}

/* Read events until `until`; returns the length of the first packet that
 * matches `want_opcode` (Command Complete / Status), printing advertising
 * reports met on the way. 0 on timeout. */
static int read_events(double until, uint16_t want_opcode, uint8_t *out, int outsz) {
    uint8_t buf[512];
    while (now() < until && !stop_now) {
        double left = until - now();
        struct timeval tv = { (long)left, (long)((left - (long)left) * 1e6) };
        fd_set r;
        FD_ZERO(&r);
        FD_SET(s, &r);
        if (select(s + 1, &r, NULL, NULL, &tv) <= 0) return 0;
        int len = read(s, buf, sizeof(buf));
        if (len < 3 || buf[0] != HCI_EVENT_PKT) continue;
        if (buf[1] == EVT_LE_META && len >= 6) {
            handle_le_meta(buf, len);
            fflush(stdout);
        } else if (want_opcode && buf[1] == EVT_CMD_COMPLETE && len >= 7
                   && (buf[4] | (buf[5] << 8)) == want_opcode) {
            memcpy(out, buf, len < outsz ? len : outsz);
            return len;
        } else if (want_opcode && buf[1] == EVT_CMD_STATUS && len >= 7
                   && (buf[5] | (buf[6] << 8)) == want_opcode) {
            memcpy(out, buf, len < outsz ? len : outsz);
            return len;
        }
    }
    return 0;
}

/* Send one LE command and wait for its status; -1 on timeout / send error. */
static int command(uint16_t opcode, const uint8_t *params, int plen) {
    uint8_t pkt[64] = { HCI_COMMAND_PKT, opcode & 0xff, opcode >> 8, (uint8_t)plen };
    memcpy(pkt + 4, params, plen);
    if (write(s, pkt, 4 + plen) < 0) return -1;
    uint8_t ev[64];
    int n = read_events(now() + 2.0, opcode, ev, sizeof(ev));
    if (n == 0) return -1;
    return ev[1] == EVT_CMD_COMPLETE ? ev[6] : ev[3];   /* Complete: status after opcode; Status: first */
}

/* Turn scanning on; 1 extended, 2 legacy, 0 already scanning (shared), -1 failed. */
static int scan_on(int *err) {
    /* Extended: own addr public, no filter policy, 1M PHY: active, 60 ms interval, 30 ms window */
    const uint8_t ext_params[] = { 0x00, 0x00, 0x01, 0x01, 0x60, 0x00, 0x30, 0x00 };
    const uint8_t ext_on[]     = { 0x01, 0x00, 0x00, 0x00, 0x00, 0x00 };   /* enable, no dup filter */
    int st = command(LE_SET_EXT_SCAN_PARAMS, ext_params, sizeof(ext_params));
    if (st == 0) {
        st = command(LE_SET_EXT_SCAN_ENABLE, ext_on, sizeof(ext_on));
        if (st == 0) return 1;
    }
    if (st == ST_DISALLOWED) return 0;
    if (st == ST_UNKNOWN_CMD) {
        const uint8_t params[] = { 0x01, 0x60, 0x00, 0x30, 0x00, 0x00, 0x00 };   /* active, 60/30 ms */
        const uint8_t on[] = { 0x01, 0x00 };
        st = command(LE_SET_SCAN_PARAMS, params, sizeof(params));
        if (st == 0) {
            st = command(LE_SET_SCAN_ENABLE, on, sizeof(on));
            if (st == 0) return 2;
        }
        if (st == ST_DISALLOWED) return 0;
    }
    *err = st;
    return -1;
}

static void scan_off(int how) {
    if (how == 1) {
        const uint8_t off[] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
        command(LE_SET_EXT_SCAN_ENABLE, off, sizeof(off));
    } else if (how == 2) {
        const uint8_t off[] = { 0x00, 0x00 };
        command(LE_SET_SCAN_ENABLE, off, sizeof(off));
    }
}

int main(int argc, char **argv) {
    double seconds = 5.0;
    int dev = 0, do_scan = 0, pos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--scan") == 0) do_scan = 1;
        else if (pos++ == 0) seconds = atof(argv[i]);
        else dev = atoi(argv[i]);
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;              /* no SA_RESTART: select() returns at once */
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    s = socket(AF_BLUETOOTH, SOCK_RAW | SOCK_CLOEXEC, BTPROTO_HCI);
    if (s < 0) { perror("socket"); return 1; }
    struct sockaddr_hci a = { AF_BLUETOOTH, (unsigned short)dev, HCI_CHANNEL_RAW };
    if (bind(s, (struct sockaddr *)&a, sizeof(a)) < 0) { perror("bind"); return 1; }
    /* event packets: Command Complete, Command Status, LE Meta */
    struct hci_filter f = { 1u << HCI_EVENT_PKT,
                            { (1u << EVT_CMD_COMPLETE) | (1u << EVT_CMD_STATUS), 1u << (EVT_LE_META - 32) }, 0 };
    if (setsockopt(s, SOL_HCI, HCI_FILTER, &f, sizeof(f)) < 0) { perror("setsockopt"); return 1; }

    int how = -2, err = 0;
    if (do_scan) {
        how = scan_on(&err);
        if (how == 1) fprintf(stderr, "scan: own-extended\n");
        else if (how == 2) fprintf(stderr, "scan: own-legacy\n");
        else if (how == 0) fprintf(stderr, "scan: shared\n");
        else fprintf(stderr, "scan: failed status %d\n", err);
    } else {
        fprintf(stderr, "scan: listen-only\n");
    }

    uint8_t unused[8];
    read_events(now() + seconds, 0, unused, sizeof(unused));
    stop_now = 0;                           /* let the disable command's reply be read */
    if (how > 0) scan_off(how);
    close(s);
    return 0;
}
