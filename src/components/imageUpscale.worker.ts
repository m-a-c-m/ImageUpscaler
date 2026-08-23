/// <reference lib="webworker" />
import { pipeline, env, RawImage } from "@huggingface/transformers";

env.allowLocalModels = false;

interface RawOut { data: Uint8ClampedArray; width: number; height: number; channels: number; }
type AnyPipe = (input: string) => Promise<RawImage | RawImage[]>;

const cache: Record<string, Promise<AnyPipe> | undefined> = {};
const cacheKey = (m: string, d: string) => `${m}::${d}`;

async function getPipe(
  modelId: string,
  device: "webgpu" | "wasm",
  dtype: string | undefined,
  onProgress: (p: number) => void,
): Promise<AnyPipe> {
  const k = cacheKey(modelId, device);
  const cached = cache[k];
  if (cached) return cached;
  const built = (async () => {
    const totals: Record<string, number> = {};
    const loaded: Record<string, number> = {};
    const p = await pipeline("image-to-image", modelId, {
      device,
      ...(dtype ? { dtype: dtype as "q8" | "fp16" } : {}),
      progress_callback: (x: { status?: string; file?: string; loaded?: number; total?: number }) => {
        if (x.status === "progress" && x.file && x.total) {
          totals[x.file] = x.total;
          loaded[x.file] = x.loaded ?? 0;
          const t = Object.values(totals).reduce((a, b) => a + b, 0);
          const l = Object.values(loaded).reduce((a, b) => a + b, 0);
          if (t > 0) onProgress(Math.min(99, Math.round((l / t) * 100)));
        }
      },
    });
    return p as unknown as AnyPipe;
  })();
  cache[k] = built;
  return built;
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data as { type: string; reqId: number; bytes: ArrayBuffer; modelId: string; device: "webgpu" | "wasm"; dtype?: string };
  if (m.type !== "run") return;
  const { reqId, bytes, modelId, device, dtype } = m;
  let url = "";
  try {
    const blob = new Blob([bytes]);
    url = URL.createObjectURL(blob);
    const pipe = await getPipe(modelId, device, dtype, (pct) => self.postMessage({ type: "progress", reqId, pct }));
    self.postMessage({ type: "processing", reqId });
    const out = await pipe(url);
    const raw = (Array.isArray(out) ? out[0] : out) as unknown as RawOut;
    if (!raw || !raw.width || !raw.data) throw new Error("empty-output");

    let data: Uint8ClampedArray;
    if (raw.channels === 4) {
      data = new Uint8ClampedArray(raw.data);
    } else {
      const px = raw.width * raw.height;
      data = new Uint8ClampedArray(px * 4);
      for (let i = 0; i < px; i++) {
        data[i * 4] = raw.data[i * 3];
        data[i * 4 + 1] = raw.data[i * 3 + 1];
        data[i * 4 + 2] = raw.data[i * 3 + 2];
        data[i * 4 + 3] = 255;
      }
    }

    self.postMessage({ type: "done", reqId, data: data.buffer, width: raw.width, height: raw.height }, [data.buffer]);
  } catch (err) {
    delete cache[cacheKey(modelId, device)];
    self.postMessage({ type: "error", reqId, message: err instanceof Error ? err.message : String(err) });
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
};
