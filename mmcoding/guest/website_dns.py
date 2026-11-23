#!/usr/bin/env python3
"""Loopback-only DNS for isolated benchmark sites, forwarding other names unchanged.

Only A/AAAA for the configured *.gitlab.<suffix> and explicit site aliases are
answered locally. Root and rv network namespaces run separate instances; this
never admits private HTTP destinations or exposes a control server to the agent.
"""
from __future__ import annotations

import argparse
import json
import socket
import socketserver
import struct
from pathlib import Path


def question(packet):
    if len(packet) < 12 or struct.unpack('!H', packet[4:6])[0] != 1:
        raise ValueError('one DNS question required')
    cursor, labels = 12, []
    while True:
        if cursor >= len(packet):
            raise ValueError('truncated DNS name')
        length = packet[cursor]
        cursor += 1
        if length == 0:
            break
        if length > 63 or cursor + length > len(packet):
            raise ValueError('compressed/invalid question name')
        labels.append(packet[cursor:cursor + length].decode('ascii').lower())
        cursor += length
    if cursor + 4 > len(packet):
        raise ValueError('truncated DNS type')
    typ, cls = struct.unpack('!HH', packet[cursor:cursor + 4])
    return '.'.join(labels), typ, cls, cursor + 4


def response(packet, config):
    name, typ, cls, end = question(packet)
    domain = config.get('gitlab_domain')
    local = name in config['hostnames'] or bool(domain and (name == domain or name.endswith('.' + domain)))
    if local and cls == 1:
        answer = b''
        if typ == 1:
            answer = b'\xc0\x0c' + struct.pack('!HHIH', 1, 1, 30, 4) + socket.inet_aton('127.0.0.1')
        # No AAAA: IPv6 loopback listener is deliberately absent.
        flags = 0x8180
        return packet[:2] + struct.pack('!HHHHH', flags, 1, int(bool(answer)), 0, 0) + packet[12:end] + answer
    for host in config['forwarders']:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(3)
                sock.sendto(packet, (host, 53))
                reply, _ = sock.recvfrom(65535)
                if reply[:2] == packet[:2]:
                    return reply
        except OSError:
            continue
    return packet[:2] + struct.pack('!HHHHH', 0x8182, 1, 0, 0, 0) + packet[12:end]


class UDP(socketserver.BaseRequestHandler):
    def handle(self):
        packet, sock = self.request
        try:
            sock.sendto(response(packet, self.server.config), self.client_address)
        except (ValueError, UnicodeError, OSError):
            pass


class TCP(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(5)
        def recv(size):
            result = b''
            while len(result) < size:
                data = self.request.recv(size - len(result))
                if not data:
                    raise OSError('DNS EOF')
                result += data
            return result
        try:
            length = struct.unpack('!H', recv(2))[0]
            reply = response(recv(length), self.server.config)
            self.request.sendall(struct.pack('!H', len(reply)) + reply)
        except (ValueError, UnicodeError, OSError):
            pass


class UDPServer(socketserver.ThreadingUDPServer):
    daemon_threads = True
    allow_reuse_address = True


class TCPServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    import threading
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--domain')
    parser.add_argument('--listen', default='127.0.0.1', choices=['127.0.0.1'])
    parser.add_argument('--port', default=53, type=int, choices=[53])
    parser.add_argument('--upstream', default='127.0.0.53')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else {
        'hostnames': [], 'gitlab_domain': args.domain, 'forwarders': [args.upstream],
    }
    with UDPServer(('127.0.0.1', 53), UDP) as udp, TCPServer(('127.0.0.1', 53), TCP) as tcp:
        udp.config = tcp.config = config
        threading.Thread(target=tcp.serve_forever, daemon=True).start()
        udp.serve_forever()


if __name__ == '__main__':
    main()
