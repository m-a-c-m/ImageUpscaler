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
  const step = Math.max(1, Math.floor(data.length / 20000));
  for (let i = 0; i < data.length; i += step) {
    if (!Number.isFinite(data[i])) return false;
  }
  return true;
}

function outputMatchesInput(data: Float32Array, nchw: Float32Array, w: number, h: number, scale: number): boolean {
  const ow = w * scale;
  const oh = h * scale;
  const opx = ow * oh;
  const step = Math.max(1, Math.floor(opx / 4000));
  let sum = 0;
  let n = 0;
  for (let oi = 0; oi < opx; oi += step) {
    const ox = oi % ow;
    const oy = (oi - ox) / ow;
    const fx = Math.max(0, Math.min(w - 1, (ox + 0.5) / scale - 0.5));
    const fy = Math.max(0, Math.min(h - 1, (oy + 0.5) / scale - 0.5));
    const x0 = Math.min(w - 1, Math.floor(fx));
    const y0 = Math.min(h - 1, Math.floor(fy));
    const x1 = Math.min(w - 1, x0 + 1);
    const y1 = Math.min(h - 1, y0 + 1);
    const ax = fx - x0;
    const ay = fy - y0;
    for (let c = 0; c < 3; c++) {
      const inPlane = c * w * h;
      const p00 = nchw[inPlane + y0 * w + x0];
      const p10 = nchw[inPlane + y0 * w + x1];
      const p01 = nchw[inPlane + y1 * w + x0];
      const p11 = nchw[inPlane + y1 * w + x1];
      const bil = p00 * (1 - ax) * (1 - ay) + p10 * ax * (1 - ay) + p01 * (1 - ax) * ay + p11 * ax * ay;
      sum += Math.abs(data[c * opx + oi] - bil);
      n++;
    }
  }
  return sum / n < (scale === 1 ? 0.25 : 0.16);
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
  return session.run({ [session.inputNames[0]]: new ort.Tensor("float32", nchw, [1, 3, h8, size]) }).then((res) => {
    const out = res[session.outputNames[0]];
    const od = out.dims as number[];
    const outW = od[3];
    const outH = od[2];
    const data = decodeMaybeFp16(out);
    if (!outputIsSane(data)) throw new Error("gpu-output-invalid");
    if (outW !== size * scale || outH !== h8 * scale) {
      throw new Error(`unexpected-output-size:${outW}x${outH}:expected:${size * scale}x${h8 * scale}`);
    }
    if (!outputMatchesInput(data, nchw, size, h8, scale)) throw new Error("output-mismatch-vs-input");
    const c = new OffscreenCanvas(outW, outH);
    const cctx = c.getContext("2d")!;
    const oimg = cctx.createImageData(outW, outH);
    const opx = outW * outH;
    for (let i = 0; i < opx; i++) {
      oimg.data[i * 4 + (bgr ? 2 : 0)] = Math.max(0, Math.min(255, data[i] * 255));
      oimg.data[i * 4 + 1] = Math.max(0, Math.min(255, data[opx + i] * 255));
      oimg.data[i * 4 + (bgr ? 0 : 2)] = Math.max(0, Math.min(255, data[2 * opx + i] * 255));
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
    const padW = raw.width - expectedW;
    const padH = raw.height - expectedH;
    if (padW < 0 || padH < 0 || padW > 32 || padH > 32) {
      throw new Error(`unexpected-output-size:${raw.width}x${raw.height}:expected:${expectedW}x${expectedH}`);
    }
    const full = hfToCanvas(raw);
    const cropped = new OffscreenCanvas(expectedW, expectedH);
    cropped.getContext("2d")!.drawImage(full, 0, 0, expectedW, expectedH, 0, 0, expectedW, expectedH);
    return cropped;
  }
  return hfToCanvas(raw);
}

self.onmessage = async (e: MessageEvent) => {
  const m = e.data as {
    type: string; reqId: number; bytes: ArrayBuffer; modelId: string; engine: "hf" | "raw";
    bgr: boolean; origin: string; device: "webgpu" | "wasm"; dtype?: string; scale: number;
    mode: "upscale" | "enhance"; tileSize: number; stride: number; ring: number; pad: number;
    preClean?: { modelId: string; bgr: boolean }; chain?: number; targetScale?: number;
  };
  if (m.type !== "run") return;
  const { reqId, bytes, modelId, engine, bgr, origin, device, dtype, scale, tileSize, stride, ring, pad } = m;
  const hops = engine === "hf" && m.chain && m.chain > 1 ? m.chain : 1;
  const effScale = Math.pow(scale, hops);
  const effTarget = m.targetScale && m.targetScale >= 1 ? m.targetScale : effScale;
  const progressCb = (pct: number) => self.postMessage({ type: "progress", reqId, pct });
  try {
    const bmp = await createImageBitmap(new Blob([bytes]));
    const w = bmp.width;
    const h = bmp.height;

    let source: CanvasImageSource = bmp;
    let preApplied = false;
    if (m.preClean && engine === "raw" && scale === 4) {
      try {
        const base = new OffscreenCanvas(w, h);
        base.getContext("2d")!.drawImage(bmp, 0, 0);
        const hasGpuPre = typeof navigator !== "undefined" && "gpu" in navigator;
        let clean: OffscreenCanvas;
        try {
          const sPre = await getRawSession(origin, m.preClean.modelId, hasGpuPre && device === "webgpu", w, progressCb);
          clean = await rawUpscale(sPre, base, 1, m.preClean.bgr);
        } catch {
          const sPre = await getRawSession(origin, m.preClean.modelId, false, w, progressCb);
          clean = await rawUpscale(sPre, base, 1, m.preClean.bgr);
        }
        const up2 = new OffscreenCanvas(w * 2, h * 2);
        const uctx = up2.getContext("2d")!;
        const srcId = clean.getContext("2d")!.getImageData(0, 0, w, h);
        const dstId = uctx.createImageData(w * 2, h * 2);
        const sd = srcId.data, dd = dstId.data;
        const uw = w * 2, uh = h * 2;
        for (let y = 0; y < uh; y++) {
          const fy = (y + 0.5) / 2 - 0.5;
          const y0 = Math.max(0, Math.min(h - 1, Math.floor(fy)));
          const y1 = Math.min(h - 1, y0 + 1);
          const ay = Math.max(0, fy - y0);
          for (let x = 0; x < uw; x++) {
            const fx = (x + 0.5) / 2 - 0.5;
            const x0 = Math.max(0, Math.min(w - 1, Math.floor(fx)));
            const x1 = Math.min(w - 1, x0 + 1);
            const ax = Math.max(0, fx - x0);
            const p00 = (y0 * w + x0) * 4, p10 = (y0 * w + x1) * 4;
            const p01 = (y1 * w + x0) * 4, p11 = (y1 * w + x1) * 4;
            const di = (y * uw + x) * 4;
            for (let c = 0; c < 3; c++) {
              dd[di + c] = sd[p00 + c] * (1 - ax) * (1 - ay) + sd[p10 + c] * ax * (1 - ay) + sd[p01 + c] * (1 - ax) * ay + sd[p11 + c] * ax * ay;
            }
            dd[di + 3] = 255;
          }
        }
        uctx.putImageData(dstId, 0, 0);
        source = up2;
        preApplied = true;
      } catch {
        source = bmp;
        preApplied = false;
      }
    }

    let runTile: (src: OffscreenCanvas, inW: number, inH: number) => Promise<OffscreenCanvas>;
    let usedDevice: "webgpu" | "wasm" = engine === "raw" ? device : device;
    let singleMode = false;

    if (engine === "raw") {
      const hasGpu = typeof navigator !== "undefined" && "gpu" in navigator;
      let useGpu = hasGpu && device === "webgpu";
      usedDevice = useGpu ? "webgpu" : "wasm";
      let gpuTried = false;
      runTile = async (src, inW, inH) => {
        try {
          const session = await getRawSession(origin, modelId, useGpu, inW, progressCb);
          const result = await rawUpscale(session, src, scale, bgr);
          gpuTried = useGpu;
          return result;
        } catch (err) {
          if (!useGpu) throw err;
          if (gpuTried) {
            throw new Error("gpu-midrun-fail");
          }
          useGpu = false;
          usedDevice = "wasm";
          progressCb(0);
          const session = await getRawSession(origin, modelId, false, inW, progressCb);
          return rawUpscale(session, src, scale, bgr);
        }
      };
    } else {
      const pipe = await getPipe(modelId, device, dtype, progressCb);
      runTile = async (src, inW, inH) => {
        let cur = src;
        for (let i = 0; i < hops; i++) {
          if (hops > 1) self.postMessage({ type: "progress", reqId, pct: singleMode ? 100 : 0, tile: i + 1, total: hops });
          cur = await runHf(pipe, cur, cur.width * scale, cur.height * scale);
        }
        void inW;
        void inH;
        return cur;
      };
    }

    const We = preApplied ? w * 2 : w;
    const He = preApplied ? h * 2 : h;
    const singleLimit = preApplied ? 512 : tileSize;
    const single = (singleLimit <= 0 || (We <= singleLimit && He <= singleLimit)) && (engine === "hf" || engine === "raw");
    let finalCanvas: OffscreenCanvas;

    if (single) {
      singleMode = true;
      self.postMessage({ type: "progress", reqId, pct: 100, tile: 1, total: 1 });
      const src = new OffscreenCanvas(We, He);
      src.getContext("2d")!.drawImage(source, 0, 0);
      finalCanvas = await runTile(src, We, He);
    } else if (engine === "raw") {
      const gw = Math.ceil(We / stride);
      const gh = Math.ceil(He / stride);
      const paddedW = ring + (gw - 1) * stride + tileSize + ring;
      const paddedH = ring + (gh - 1) * stride + tileSize + ring;
      const total = gw * gh;
      let done = 0;

      const padded = new OffscreenCanvas(paddedW, paddedH);
      const pctx = padded.getContext("2d")!;
      pctx.drawImage(source, ring, ring);
      if (ring > 0) {
        if (We < paddedW - ring) pctx.drawImage(source, We - 1, 0, 1, He, ring + We, ring, paddedW - ring - We, He);
        if (He < paddedH - ring) pctx.drawImage(source, 0, He - 1, We, 1, ring, ring + He, We, paddedH - ring - He);
        if (We < paddedW - ring && He < paddedH - ring) pctx.drawImage(source, We - 1, He - 1, 1, 1, ring + We, ring + He, paddedW - ring - We, paddedH - ring - He);
      }

      const out = new OffscreenCanvas(gw * stride * effScale, gh * stride * effScale);
      const octx = out.getContext("2d")!;
      const keepW = stride * effScale;
      const keepH = stride * effScale;
      const crop = ring * effScale;

      for (let j = 0; j < gh; j++) {
        for (let i = 0; i < gw; i++) {
          const ox = ring + i * stride;
          const oy = ring + j * stride;
          const tmp = new OffscreenCanvas(tileSize, tileSize);
          const tctx = tmp.getContext("2d")!;
          tctx.drawImage(padded, ox, oy, tileSize, tileSize, 0, 0, tileSize, tileSize);

          const tileOut = await runTile(tmp, tileSize, tileSize);
          octx.drawImage(tileOut, crop, crop, keepW, keepH, i * stride * effScale, j * stride * effScale, keepW, keepH);

          done++;
          self.postMessage({ type: "progress", reqId, pct: 100, tile: done, total });
        }
      }

      let composed: OffscreenCanvas;
      if (preApplied) {
        composed = new OffscreenCanvas(w * effTarget, h * effTarget);
        const cctx = composed.getContext("2d")!;
        cctx.imageSmoothingEnabled = true;
        cctx.imageSmoothingQuality = "high";
        cctx.drawImage(out, 0, 0, out.width, out.height, 0, 0, w * effTarget, h * effTarget);
      } else {
        composed = new OffscreenCanvas(w * effTarget, h * effTarget);
        const cctx = composed.getContext("2d")!;
        cctx.imageSmoothingEnabled = true;
        cctx.imageSmoothingQuality = "high";
        cctx.drawImage(out, 0, 0, w * effScale, h * effScale, 0, 0, w * effTarget, h * effTarget);
      }
      finalCanvas = composed;
    } else {
      const tw = Math.min(tileSize, w);
      const th = Math.min(tileSize, h);
      const cols = Math.ceil(w / tw);
      const rows = Math.ceil(h / th);
      const total = cols * rows;
      let done = 0;

      const out = new OffscreenCanvas(w * effScale, h * effScale);
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
            (sx - px) * effScale, (sy - py) * effScale, tw * effScale, th * effScale,
            sx * effScale, sy * effScale, tw * effScale, th * effScale
          );

          done++;
          self.postMessage({ type: "progress", reqId, pct: 100, tile: done, total });
        }
      }
      finalCanvas = out;
    }

    const targetW = w * effTarget;
    const targetH = h * effTarget;
    if (finalCanvas.width !== targetW || finalCanvas.height !== targetH) {
      const small = new OffscreenCanvas(targetW, targetH);
      const sctx = small.getContext("2d")!;
      sctx.imageSmoothingEnabled = true;
      sctx.imageSmoothingQuality = "high";
      sctx.drawImage(finalCanvas, 0, 0, targetW, targetH);
      finalCanvas = small;
    }

    const fctx = finalCanvas.getContext("2d")!;
    const fw = finalCanvas.width;
    const fh = finalCanvas.height;
    const imageData = fctx.getImageData(0, 0, fw, fh);
    const buf = imageData.data.buffer as ArrayBuffer;
    self.postMessage({ type: "done", reqId, data: buf, width: fw, height: fh, device: usedDevice, pre: preApplied }, [buf]);
  } catch (err) {
    if (engine === "hf") delete hfCache[`${modelId}::${device}`];
    for (const k of Object.keys(ortSessions)) if (k.includes(modelId)) delete ortSessions[k];
    self.postMessage({ type: "error", reqId, message: err instanceof Error ? err.message : String(err) });
  }
};
