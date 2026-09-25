---
name: OpenAPI browser types
description: Why this workspace's generated API libraries need browser library types when contracts include multipart uploads.
---

Generated Orval clients and Zod schemas can reference `Blob`, `Headers.entries()`, and iterable browser interfaces even when the API server itself is Node-based.

**Why:** A multipart audio upload contract generated browser-facing types that failed the shared library build until the relevant `lib` entries were explicit.

**How to apply:** When adding multipart, Blob, or browser-header schemas to `lib/api-spec/openapi.yaml`, keep `dom` and `dom.iterable` enabled in the generated client and Zod library TypeScript configs, then rerun codegen.