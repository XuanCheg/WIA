/**
 * Run: node tests/harness_integration/test_esi_native_editor_transport.js
 * Verify the real iframe using noVNC 1.7's host-empty URL resolution.
 * This test opens no network connections and starts no services.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../../src/wia/runtime/web/native_editor/index.html'), 'utf8');
const iframe = html.match(/<iframe\b[^>]*\bid="native"[^>]*\bsrc="([^"]+)"/);
assert.ok(iframe, 'The native GUI iframe must exist.');
const src = iframe[1].replaceAll('&amp;', '&');

function transportURL(pageURL, iframeSrc) {
    const iframeURL = new URL(iframeSrc, pageURL);
    const query = iframeURL.searchParams;
    assert.equal(query.get('autoconnect'), '1');
    assert.equal(query.get('resize'), 'scale');
    assert.equal(query.get('host'), null, 'Use the browser host, not a fixed remote host.');
    assert.equal(query.get('port'), null, 'Use the forwarded browser port.');
    const socket = new URL(query.get('path'), iframeURL);
    socket.protocol = iframeURL.protocol === 'https:' ? 'wss:' : 'ws:';
    return socket;
}

for (const [page, expected] of [
    ['http://localhost:8615/', 'ws://localhost:8615/websockify'],
    ['http://localhost:18615/?scene_id=core_ab', 'ws://localhost:18615/websockify'],
    ['https://editor.example.test:9443/', 'wss://editor.example.test:9443/websockify'],
]) {
    assert.equal(transportURL(page, src).href, expected);
}

// The former relative query resolves under /novnc/ and must fail the root check.
const oldSrc = '/novnc/vnc.html?autoconnect=1&resize=scale&path=websockify';
assert.equal(transportURL('http://localhost:8615/', oldSrc).pathname, '/novnc/websockify');
assert.notEqual(transportURL('http://localhost:8615/', oldSrc).pathname, '/websockify');
console.log('Native editor transport: three browser origins resolve root /websockify; old relative path rejected.');
