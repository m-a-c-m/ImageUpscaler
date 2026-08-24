/// <reference lib="webworker" />
import { pipeline, env } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";
import { fetchArrayBufferCached } from "./cachedFetch";

env.allowLocalModels = false;

interface RawOut { data: Uint8ClampedArray; width: number; height: number; channels: number; }
type AnyPipe = (input: string) => Promise<unknown>;

const hfCache: Record<string, Promise<AnyPipe> | undefined> = {};
const ortSessions: Record<string, Promise<ort.InferenceSession> | undefined> = {};

async function getPipe(
  modelId: string,
  device: "webgpu" | "wasm",
  dtype: string | undefined,
  onProgress: (p: number) => void,
): Promise<AnyPipe> {
  const k = `${modelId}::${device}`;
  const cached = hfCache[k];
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
  hfCache[k] = built;
  return built;
}

async function getRawSession(
  origin: string,
  modelUrl: string,
  useGpu: boolean,
  inputSize: number,
  onProgress: (p: number) => void,
): Promise<ort.InferenceSession> {
  const key = `${useGpu ? "gpu" : "cpu"}::${modelUrl}::${inputSize}`;
  const cached = ortSessions[key];
  if (cached) return cached;
  const built = (async () => {
    const wasmUrl = useGpu ? `${origin}/ort/ort-wasm-simd-threaded.jsep.wasm` : `${origin}/ort/ort-wasm-simd-threaded.wasm`;
    const mjsUrl = useGpu ? `${origin}/ort/ort-wasm-simd-threaded.jsep.mjs` : `${origin}/ort/ort-wasm-simd-threaded.mjs`;
    const [wasmBinary, modelBuffer] = await Promise.all([
      fetchArrayBufferCached(wasmUrl, "upscaler-ort-v1"),
      fetchArrayBufferCached(modelUrl, "upscaler-models-v1", (loaded, total) => onProgress(Math.min(99, Math.round((loaded / total) * 100)))),
    ]);
    ort.env.wasm.wasmBinary = new Uint8Array(wasmBinary);
    ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
    return ort.InferenceSession.create(new Uint8Array(modelBuffer), {
      executionProviders: [useGpu ? "webgpu" : "wasm"],
    });
  })();
  ortSessions[key] = built;
  built.catch(() => {
    delete ortSessions[key];
  });
  return built;
}

function decodeMaybeFp16(t: ort.Tensor): Float32Array {
  if (t.type === "float16") {
    const u16 = t.data as unknown as Uint16Array;
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const s = (h & 0x8000) >> 15;
      const e = (h & 0x7c00) >> 10;
      const f = h & 0x03ff;
      let val: number;
      if (e === 0) val = (f / 1024) * 2 ** -14;
      else if (e === 31) val = f === 0 ? Infinity : NaN;
      else val = (1 + f / 1024) * 2 ** (e - 15);
      out[i] = s ? -val : val;
    }
    return out;
  }
  return t.data as unknown as Float32Array;
}

function outputIsSane(data: Float32Array): boolean {
  let bad = 0;
  let sum = 0;
  let sumSq = 0;
  const n = data.length;
  const step = Math.max(1, Math.floor(n / 20000));
  let count = 0;
  for (let i = 0; i < n; i += step) {
    const v = data[i];
    count++;
    if (Number.isNaN(v) || !Number.isFinite(v)) bad++;
    else {
      if (v < -1 || v > 2) bad++;
      sum += v;
      sumSq += v * v;
    }
  }
  if (bad > 0) return false;
  const mean = sum / count;
  const variance = sumSq / count - mean * mean;
  return variance > 0.0002;
}

function rawUpscale(
  session: ort.InferenceSession,
  src: OffscreenCanvas,
  scale: number,
  bgr: boolean,
): Promise<OffscreenCanvas> {
  const size = src.width;
  const h8 = src.height;
  const ctx = src.getContext("2d")!;
  const img = ctx.getImageData(0, 0, size, h8);
  const px = size * h8;
  const nchw = new Float32Array(3 * px);
  for (let i = 0; i < px; i++) {
    nchw[i] = img.data[i * 4 + (bgr ? 2 : 0)] / 255;
    nchw[px + i] = img.data[i * 4 + 1] / 255;
    nchw[2 * px + i] = img.data[i * 4 + (bgr ? 0 : 2)] / 255;
  }
  return session.run({ [session.inputNames[0]]: new ort.Tensor("float32", nchw, [1, 3, size, h8]) }).then((res) => {
    const out = res[session.outputNames[0]];
    const od = out.dims as number[];
    const outW = od[3];
    const outH = od[2];
    const data = decodeMaybeFp16(out);
    if (!outputIsSane(data)) throw new Error("gpu-output-invalid");
    const c = new OffscreenCanvas(outW, outH);
    const cctx = c.getContext("2d")!;
    const oimg = cctx.createImageData(outW, outH);
    const opx = outW * outH;
    for (let i = 0; i < opx; i++) {
      oimg.data[i * 4 + (bgr ? 2 : 0)] = Math.max(0, Math.min(255, data[i] * 255));
      oimg.data[i * 4 + 1] = Math.max(0, Math.min(255, data[px + i] * 255));
      oimg.data[i * 4 + (bgr ? 0 : 2)] = Math.max(0, Math.min(255, data[2 * px + i] * 255));
      oimg.data[i * 4 + 3] = 255;
    }
    cctx.putImageData(oimg, 0, 0);
    return c;
  });
}

function hfToCanvas(raw: RawOut): OffscreenCanvas {
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

async function runHf(pipe: AnyPipe, canvas: OffscreenCanvas, expectedW: number, expectedH: number): Promise<OffscreenCanvas> {
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
  return hfToCanvas(raw);
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data as {
    type: string; reqId: number; bytes: ArrayBuffer; modelId: string; engine: "hf" | "raw";
    bgr: boolean; origin: string; device: "webgpu" | "wasm"; dtype?: string; scale: number;
    mode: "upscale" | "enhance"; tileSize: number; stride: number; ring: number; pad: number;
  };
  if (m.type !== "run") return;
  const { reqId, bytes, modelId, engine, bgr, origin, device, dtype, scale, mode, tileSize, stride, ring, pad } = m;
  try {
    const bmp = await createImageBitmap(new Blob([bytes]));
    const w = bmp.width;
    const h = bmp.height;

    let runTile: (src: OffscreenCanvas, inW: number, inH: number) => Promise<OffscreenCanvas>;
    const progressCb = (pct: number) => self.postMessage({ type: "progress", reqId, pct });

    if (engine === "raw") {
      const hasGpu = typeof navigator !== "undefined" && "gpu" in navigator;
      let useGpu = hasGpu;
      runTile = async (src, inW, inH) => {
        try {
          const session = await getRawSession(origin, modelId, useGpu, inW, progressCb);
          return await rawUpscale(session, src, scale, bgr);
        } catch (err) {
          if (!useGpu) throw err;
          useGpu = false;
          progressCb(0);
          const session = await getRawSession(origin, modelId, false, inW, progressCb);
          return rawUpscale(session, src, scale, bgr);
        }
      };
    } else {
      const pipe = await getPipe(modelId, device, dtype, progressCb);
      runTile = (src, inW, inH) => runHf(pipe, src, inW * scale, inH * scale);
    }

    const single = engine === "hf" && (tileSize <= 0 || (w <= tileSize && h <= tileSize));
    let finalCanvas: OffscreenCanvas;

    if (single) {
      self.postMessage({ type: "progress", reqId, pct: 100, tile: 1, total: 1 });
      const src = new OffscreenCanvas(w, h);
      src.getContext("2d")!.drawImage(bmp, 0, 0);
      finalCanvas = await runTile(src, w, h);
    } else if (engine === "raw") {
      const gw = Math.ceil(w / stride);
      const gh = Math.ceil(h / stride);
      const paddedW = ring + (gw - 1) * stride + tileSize + ring;
      const paddedH = ring + (gh - 1) * stride + tileSize + ring;
      const total = gw * gh;
      let done = 0;

      const padded = new OffscreenCanvas(paddedW, paddedH);
      const pctx = padded.getContext("2d")!;
      pctx.drawImage(bmp, ring, ring);
      if (ring > 0) {
        if (w < paddedW - ring) pctx.drawImage(bmp, w - 1, 0, 1, h, ring + w, ring, paddedW - ring - w, h);
        if (h < paddedH - ring) pctx.drawImage(bmp, 0, h - 1, w, 1, ring, ring + h, w, paddedH - ring - h);
        if (w < paddedW - ring && h < paddedH - ring) pctx.drawImage(bmp, w - 1, h - 1, 1, 1, ring + w, ring + h, paddedW - ring - w, paddedH - ring - h);
      }

      const out = new OffscreenCanvas(gw * stride * scale, gh * stride * scale);
      const octx = out.getContext("2d")!;
      const keepW = stride * scale;
      const keepH = stride * scale;
      const crop = ring * scale;

      for (let j = 0; j < gh; j++) {
        for (let i = 0; i < gw; i++) {
          const ox = ring + i * stride;
          const oy = ring + j * stride;
          const tmp = new OffscreenCanvas(tileSize, tileSize);
          const tctx = tmp.getContext("2d")!;
          tctx.drawImage(padded, ox, oy, tileSize, tileSize, 0, 0, tileSize, tileSize);

          const tileOut = await runTile(tmp, tileSize, tileSize);
          octx.drawImage(tileOut, crop, crop, keepW, keepH, i * stride * scale, j * stride * scale, keepW, keepH);

          done++;
          self.postMessage({ type: "progress", reqId, pct: 100, tile: done, total });
        }
      }

      const cropped = new OffscreenCanvas(w * scale, h * scale);
      cropped.getContext("2d")!.drawImage(out, 0, 0, w * scale, h * scale, 0, 0, w * scale, h * scale);
      finalCanvas = cropped;
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

          const tileOut = await runTile(tmp, pw, ph);
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

    if (mode === "enhance" && scale !== 1) {
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
    if (engine === "hf") delete hfCache[`${modelId}::${device}`];
    for (const k of Object.keys(ortSessions)) if (k.includes(modelId)) delete ortSessions[k];
    self.postMessage({ type: "error", reqId, message: err instanceof Error ? err.message : String(err) });
  }
};
