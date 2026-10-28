# Security Policy

## Supported versions

silgi is pre-1.0. Security fixes go into the latest release only.

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Use GitHub's private
reporting instead: open the repository's **Security** tab and choose
**Report a vulnerability**. Include what you found, how to reproduce it and the
silgi version. You will get a response within a few days; we will agree on a
disclosure date with you once a fix is ready.

## Scope and design notes

silgi is a local tool, and its security posture follows from that:

* The server binds to `127.0.0.1` by default. **Do not expose it to a network
  you do not trust**: it has no authentication, because it is meant to be used by
  the person running it. If you pass `--host 0.0.0.0` you are responsible for
  protecting the port.
* silgi never makes outbound network requests at runtime; fonts and assets are
  bundled. The test suite runs with outbound sockets blocked.
* Image decoding uses Pillow, OpenCV and tifffile. Keep them up to date; most
  realistic risks are in image parsers. Uploads are size-limited and Pillow's
  decompression-bomb protection is enabled (the pixel limit is raised to 400 MP
  for legitimate very large images).
* Exports drop EXIF by default so location and device details are not carried
  into shared files.

Reports about missing authentication on a deliberately exposed server are out of
scope; reports about path traversal, memory-safety issues triggered by crafted
images, or the server reaching the network on its own are in scope.
