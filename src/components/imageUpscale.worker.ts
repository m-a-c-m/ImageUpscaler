/// <reference lib="webworker" />
import { pipeline, env } from "@huggingface/transformers";

env.allowLocalModels = false;

interface RawOut { data: Uint8ClampedArray; width: number; height: number; channels: number; }
type AnyPipe = (input: string) => Promise<unknown>;

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

function rawToCanvas(raw: RawOut): OffscreenCanvas {
  const c = new OffscreenCanvas(raw.width, raw.height);
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(raw.width, raw.height);
  if (raw.channels === 4) {
    img.data.set(raw.data);
  } else {
    const px = raw.width * raw.height;
    for (let i = 0; i < px; i++) {
      img.data[i * 4] = raw.data[i * 3];
      img.data[i * 4 + 1] = raw.data[i * 3 + 1];
      img.data[i * 4 + 2] = raw.data[i * 3 + 2];
      img.data[i * 4 + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

async function runModel(pipe: AnyPipe, canvas: OffscreenCanvas, expectedW: number, expectedH: number): Promise<OffscreenCanvas> {
  const blob = await canvas.convertToBlob({ type: "image/png" });
  const url = URL.createObjectURL(blob);
  let raw: RawOut;
  try {
    const res = await pipe(url);
    raw = (Array.isArray(res) ? res[0] : res) as unknown as RawOut;
  } finally {
    URL.revokeObjectURL(url);
  }
  if (!raw || !raw.width || !raw.data) throw new Error("empty-output");
  if (raw.width !== expectedW || raw.height !== expectedH) {
    throw new Error(`unexpected-output-size:${raw.width}x${raw.height}:expected:${expectedW}x${expectedH}`);
  }
  return rawToCanvas(raw);
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data as {
    type: string; reqId: number; bytes: ArrayBuffer; modelId: string;
    device: "webgpu" | "wasm"; dtype?: string; scale: number;
    mode: "upscale" | "enhance"; tileSize: number; pad: number;
  };
  if (m.type !== "run") return;
  const { reqId, bytes, modelId, device, dtype, scale, mode, tileSize, pad } = m;
  try {
    const pipe = await getPipe(modelId, device, dtype, (pct) =>
      self.postMessage({ type: "progress", reqId, pct })
    );

    const bmp = await createImageBitmap(new Blob([bytes]));
    const w = bmp.width;
    const h = bmp.height;

    const single = tileSize <= 0 || (w <= tileSize && h <= tileSize);
    let finalCanvas: OffscreenCanvas;

    if (single) {
      self.postMessage({ type: "progress", reqId, pct: 100, tile: 1, total: 1 });
      const src = new OffscreenCanvas(w, h);
      src.getContext("2d")!.drawImage(bmp, 0, 0);
      finalCanvas = await runModel(pipe, src, w * scale, h * scale);
    } else {
      const tw = Math.min(tileSize, w);
      const th = Math.min(tileSize, h);
      const cols = Math.ceil(w / tw);
      const rows = Math.ceil(h / th);
      const total = cols * rows;
      let done = 0;

      const out = new OffscreenCanvas(w * scale, h * scale);
      const octx = out.getContext("2d")!;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const sx = c === cols - 1 ? Math.max(0, w - tw) : c * tw;
          const sy = r === rows - 1 ? Math.max(0, h - th) : r * th;
          const px = Math.max(0, sx - pad);
          const py = Math.max(0, sy - pad);
          const pw = Math.min(tw + pad * 2, w - px);
          const ph = Math.min(th + pad * 2, h - py);

          const tmp = new OffscreenCanvas(pw, ph);
          const tctx = tmp.getContext("2d")!;
          tctx.drawImage(bmp, px - sx, py - sy);

          const tileOut = await runModel(pipe, tmp, pw * scale, ph * scale);
          octx.drawImage(
            tileOut,
            (sx - px) * scale, (sy - py) * scale, tw * scale, th * scale,
            sx * scale, sy * scale, tw * scale, th * scale
          );

          done++;
          self.postMessage({ type: "progress", reqId, pct: 100, tile: done, total });
        }
      }
      finalCanvas = out;
    }

    if (mode === "enhance") {
      const small = new OffscreenCanvas(w, h);
      const sctx = small.getContext("2d")!;
      sctx.imageSmoothingEnabled = true;
      sctx.imageSmoothingQuality = "high";
      sctx.drawImage(finalCanvas, 0, 0, w, h);
      finalCanvas = small;
    }

    const fctx = finalCanvas.getContext("2d")!;
    const fw = finalCanvas.width;
    const fh = finalCanvas.height;
    const imageData = fctx.getImageData(0, 0, fw, fh);
    const buf = imageData.data.buffer as ArrayBuffer;
    self.postMessage({ type: "done", reqId, data: buf, width: fw, height: fh }, [buf]);
  } catch (err) {
    delete cache[cacheKey(modelId, device)];
    self.postMessage({ type: "error", reqId, message: err instanceof Error ? err.message : String(err) });
  }
};
