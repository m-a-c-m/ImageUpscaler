// Plain fetch() relies on the server's Cache-Control header for reuse across
// visits. Both of our own model hosts undercut that badly for how large
// these files are: GitHub's raw.githubusercontent.com sends max-age=300 (5
// minutes) on the HTDemucs model chunks, and Vercel serves /public/ static
// assets with max-age=0, must-revalidate by default — so every VAD/Whisper
// run and every stem-separation re-sent a revalidation request (or a full
// re-download) for tens to hundreds of MB, even though nothing changed.
// @huggingface/transformers avoids this for its own models by caching into
// the Cache Storage API explicitly (bucket "transformers-cache") instead of
// trusting HTTP cache headers — this does the same thing for the two tools
// that talk to onnxruntime-web directly instead of through that pipeline.
//
// Cache Storage entries persist until deleted, independent of Cache-Control,
// which is exactly why cache-name is date/version-stamped: if a hosted model
// is ever replaced at the same URL, a stale cached copy would otherwise be
// served forever. Bump the version suffix (and ideally the file's path too)
// whenever the underlying bytes change.
export async function fetchArrayBufferCached(
  url: string,
  cacheName: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(url);
  if (cached) {
    const buf = await cached.arrayBuffer();
    onProgress?.(buf.byteLength, buf.byteLength);
    return buf;
  }

  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`fetch-failed:${url}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  // Cache a clone before consuming the body ourselves for progress
  // tracking — best-effort, a full storage quota shouldn't break the tool.
  cache.put(url, res.clone()).catch(() => {});

  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    onProgress?.(loaded, total || loaded);
  }
  const merged = new Uint8Array(loaded);
  let off = 0;
  for (const p of parts) { merged.set(p, off); off += p.length; }
  return merged.buffer;
}
