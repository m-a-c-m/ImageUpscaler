"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FiUploadCloud, FiDownload, FiLoader, FiX,
  FiAlertCircle, FiCpu, FiZap, FiClock, FiMaximize2, FiFileText,
} from "react-icons/fi";

interface Props { locale?: string; }

type Stage = "idle" | "loadingModel" | "processing" | "done" | "error";
type Mode = "upscale" | "logo" | "enhance";
type ModelKey = "x4-fiel" | "x4-plksr" | "x4-esrgan" | "x2-light" | "dejpeg" | "denoise";

interface ModelDef {
  id: string;
  engine: "hf" | "raw";
  bgr: boolean;
  scale: 1 | 2 | 4;
  tileSize: number;
  stride: number;
  ring: number;
  pad: number;
  maxSide: number;
  shotLimit: number;
  mb: number;
  es: string;
  en: string;
  descEs: string;
  descEn: string;
}

const NES = "https://huggingface.co/nesaorg";

const MODELS: Record<ModelKey, ModelDef> = {
  "x4-fiel": { id: `${NES}/4xLSDIRCompactN_fp32_opset17/resolve/main/4xLSDIRCompactN_fp32_opset17.onnx`, engine: "raw", bgr: false, scale: 4, tileSize: 256, stride: 192, ring: 32, pad: 0, maxSide: 4096, shotLimit: 0, mb: 2, es: "Fiel — sin alterar tu foto", en: "Faithful — keeps your photo as is", descEs: "Amplifica con IA reproduciendo la imagen tal cual, sin inventar detalle. Rápido y ligero.", descEn: "AI upscaling that reproduces the image exactly, without inventing detail. Fast and lightweight." },
  "x4-plksr": { id: `${NES}/4xNomosWebPhoto_RealPLKSR_fp32_opset17/resolve/main/4xNomosWebPhoto_RealPLKSR_fp32_opset17.onnx`, engine: "raw", bgr: false, scale: 4, tileSize: 256, stride: 192, ring: 32, pad: 0, maxSide: 1024, shotLimit: 0, mb: 28, es: "Con realce estético", en: "Enhanced look", descEs: "Suaviza imperfecciones y embellece, pero altera más la foto original.", descEn: "Smooths imperfections and beautifies, but alters the original more." },
  "x4-esrgan": { id: `${NES}/4xNomosWebPhoto_esrgan_fp32_opset17/resolve/main/4xNomosWebPhoto_esrgan_fp32_opset17.onnx`, engine: "raw", bgr: false, scale: 4, tileSize: 256, stride: 192, ring: 32, pad: 0, maxSide: 1024, shotLimit: 0, mb: 64, es: "Máxima calidad", en: "Max quality", descEs: "Más detalle en texturas, pero más lento y altera más.", descEn: "More texture detail, but slower and alters more." },
  "x2-light": { id: "Xenova/swin2SR-lightweight-x2-64", engine: "hf", bgr: false, scale: 2, tileSize: 256, stride: 256, ring: 32, pad: 32, maxSide: 2048, shotLimit: 1024, mb: 8, es: "Logos y capturas", en: "Logos and screenshots", descEs: "El motor para gráficos: logos, dibujos y capturas. Llega a ×4 procesando en 2 pasos.", descEn: "The engine for graphics: logos, drawings and screenshots. Reaches ×4 in 2 steps." },
  dejpeg: { id: `${NES}/1xDeJPG_realplksr_otf_60_fp32_opset17/resolve/main/1xDeJPG_realplksr_otf_60_fp32_opset17.onnx`, engine: "raw", bgr: false, scale: 1, tileSize: 256, stride: 192, ring: 32, pad: 0, maxSide: 4096, shotLimit: 0, mb: 28, es: "Bloques o manchas de compresión", en: "Compression blocks or stains", descEs: "Fotos de WhatsApp, redes sociales o reenviadas muchas veces.", descEn: "Photos from WhatsApp, social media or forwarded many times." },
  denoise: { id: `${NES}/1xDeNoise_realplksr_otf_fp32/resolve/main/1xDeNoise_realplksr_otf_fp32.onnx`, engine: "raw", bgr: false, scale: 1, tileSize: 256, stride: 192, ring: 32, pad: 0, maxSide: 4096, shotLimit: 0, mb: 28, es: "Grano o ruido", en: "Grain or noise", descEs: "Fotos nocturnas, con poca luz o muy granuladas.", descEn: "Night shots, low-light or very grainy photos." },
};

const MODELS_BY_MODE: Record<Mode, ModelKey[]> = {
  upscale: ["x4-fiel", "x4-plksr", "x4-esrgan"],
  logo: ["x2-light"],
  enhance: ["dejpeg", "denoise"],
};

interface Attempt { device: "webgpu" | "wasm"; dtype?: string; }
let reqCounter = 0;

interface WorkerResult { data: Uint8ClampedArray; width: number; height: number; }

function runInWorker(
  worker: Worker,
  payload: { bytes: ArrayBuffer; modelId: string; engine: string; bgr: boolean; origin: string; device: string; dtype?: string; scale: number; mode: Mode; tileSize: number; stride: number; ring: number; pad: number; preClean?: { modelId: string; bgr: boolean }; chain?: number; targetScale?: number },
  onProgress: (pct: number, tile?: number, total?: number) => void,
): Promise<WorkerResult & { device?: string; pre?: boolean }> {
  return new Promise((resolve, reject) => {
    const reqId = ++reqCounter;
    const buf = payload.bytes.slice(0);
    const handler = (e: MessageEvent) => {
      const m = e.data as { type: string; reqId: number; pct?: number; tile?: number; total?: number; data?: ArrayBuffer; width?: number; height?: number; message?: string; device?: string; pre?: boolean };
      if (m.reqId !== reqId) return;
      if (m.type === "progress") onProgress(m.pct ?? 0, m.tile, m.total);
      else if (m.type === "done") {
        worker.removeEventListener("message", handler);
        resolve({ data: new Uint8ClampedArray(m.data!), width: m.width!, height: m.height!, device: m.device, pre: m.pre });
      } else if (m.type === "error") {
        worker.removeEventListener("message", handler);
        reject(new Error(m.message ?? "error"));
      }
    };
    worker.addEventListener("message", handler);
    try {
      worker.postMessage({ type: "run", reqId, bytes: buf, modelId: payload.modelId, engine: payload.engine, bgr: payload.bgr, origin: payload.origin, device: payload.device, dtype: payload.dtype, scale: payload.scale, mode: payload.mode, tileSize: payload.tileSize, stride: payload.stride, ring: payload.ring, pad: payload.pad, preClean: payload.preClean, chain: payload.chain, targetScale: payload.targetScale }, [buf]);
    } catch (err) {
      worker.removeEventListener("message", handler);
      reject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

async function downscaleToCap(file: File, maxSide: number): Promise<{ bytes: ArrayBuffer; url: string; width: number; height: number; srcW: number; srcH: number; capped: boolean }> {
  const bmp = await createImageBitmap(file);
  const w = bmp.width;
  const h = bmp.height;
  const scale = Math.min(1, maxSide / Math.max(w, h));
  if (scale >= 1) {
    const buf = await file.arrayBuffer();
    bmp.close();
    return { bytes: buf, url: URL.createObjectURL(file), width: w, height: h, srcW: w, srcH: h, capped: false };
  }
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);
  const c = document.createElement("canvas");
  c.width = nw;
  c.height = nh;
  const ctx = c.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, nw, nh);
  bmp.close();
  const blob = await new Promise<Blob>((res) => c.toBlob((b) => res(b!), "image/png"));
  const buf = await blob.arrayBuffer();
  return { bytes: buf, url: URL.createObjectURL(blob), width: nw, height: nh, srcW: w, srcH: h, capped: true };
}

export default function ImageUpscaler({ locale = "es" }: Props) {
  const isEs = locale === "es";

  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [tileInfo, setTileInfo] = useState("");
  const [error, setError] = useState("");
  const [gpuDied, setGpuDied] = useState(false);
  const [forceCpu, setForceCpu] = useState(false);
  const [mode, setMode] = useState<Mode>("upscale");
  const [modelKey, setModelKey] = useState<ModelKey>("x4-fiel");
  const [targetScale, setTargetScale] = useState<2 | 3 | 4>(2);
  const [origUrl, setOrigUrl] = useState("");
  const [resultUrl, setResultUrl] = useState("");
  const [origDims, setOrigDims] = useState({ w: 0, h: 0 });
  const [resultDims, setResultDims] = useState({ w: 0, h: 0 });
  const [srcDims, setSrcDims] = useState({ w: 0, h: 0 });
  const [usedLabel, setUsedLabel] = useState("");
  const [resultPre, setResultPre] = useState(false);
  const [elapsed, setElapsed] = useState("");
  const [capped, setCapped] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [hasGpu, setHasGpu] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [divider, setDivider] = useState(50);
  const [showTech, setShowTech] = useState(false);

  const workerRef = useRef<Worker | null>(null);
  const fileRef = useRef<File | null>(null);
  const compareRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const cancelRef = useRef(false);

  const spawnWorker = useCallback(() => {
    workerRef.current?.terminate();
    const w = new Worker(new URL("./imageUpscale.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = w;
  }, []);

  useEffect(() => {
    setIsMobile(window.matchMedia("(max-width: 639px)").matches);
    setHasGpu(typeof navigator !== "undefined" && "gpu" in navigator);
    spawnWorker();
    return () => workerRef.current?.terminate();
  }, [spawnWorker]);

  const model = MODELS[modelKey];
  const availableKeys = MODELS_BY_MODE[mode];

  useEffect(() => {
    const keys = MODELS_BY_MODE[mode];
    setModelKey((k) => (keys.includes(k) ? k : keys[0]));
  }, [mode]);

  useEffect(() => {
    if (stage !== "idle" && stage !== "error" && stage !== "done") return;
    setError("");
  }, [modelKey, mode]);

  const run = useCallback(
    async (file: File) => {
      setStage("loadingModel");
      setProgress(0);
      setTileInfo("");
      setError("");
      setResultUrl("");
      setGpuDied(false);
      setResultPre(false);

      const prepared = await downscaleToCap(file, model.maxSide);
      setOrigUrl(prepared.url);
      setOrigDims({ w: prepared.width, h: prepared.height });
      setSrcDims({ w: prepared.srcW, h: prepared.srcH });
      setCapped(prepared.capped);
      fileRef.current = file;

      const colsT = Math.ceil(prepared.width / model.stride);
      const rowsT = Math.ceil(prepared.height / model.stride);
      const nTiles = colsT * rowsT;

      if (model.engine === "raw" && !hasGpu && !forceCpu && nTiles > 60) {
        setError(
          isEs
            ? `Tu navegador no tiene WebGPU y esta imagen necesita ${nTiles} pasos de procesamiento (más de una hora). Prueba con una imagen más pequeña o usa Chrome/Edge.`
            : `Your browser has no WebGPU and this image needs ${nTiles} processing steps (over an hour). Try a smaller image or use Chrome/Edge.`
        );
        setStage("error");
        return;
      }

      if (mode === "logo" && Math.max(prepared.width, prepared.height) > 1400) {
        setError(
          isEs
            ? "Esta imagen es muy grande para el motor de gráficos ×4 (tardaría más de 20 minutos). Reduce el tamaño de la imagen primero o usa el modo foto."
            : "This image is too large for the ×4 graphics engine (it would take over 20 minutes). Resize it first or use the photo mode."
        );
        setStage("error");
        return;
      }

      const attempts: Attempt[] = model.engine === "raw"
        ? forceCpu || !hasGpu ? [{ device: "wasm" }] : [{ device: "webgpu" }]
        : forceCpu
          ? [{ device: "wasm", dtype: "q8" }]
          : [
              { device: "wasm", dtype: "q8" },
              { device: "webgpu", dtype: "fp16" },
            ];

      let lastError = "";
      cancelRef.current = false;
      for (const attempt of attempts) {
        if (cancelRef.current) break;
        try {
          const gpuHere = typeof navigator !== "undefined" && "gpu" in navigator;
          if (attempt.device === "webgpu" && !gpuHere) continue;
          const worker = workerRef.current;
          if (!worker) throw new Error("no-worker");

          const t0 = performance.now();
          const usePreClean = mode === "upscale" && model.engine === "raw" && model.scale === 4 && Math.max(prepared.width, prepared.height) < 320;
          const result = await runInWorker(
            worker,
            { bytes: prepared.bytes, modelId: model.id, engine: model.engine, bgr: model.bgr, origin: window.location.origin, device: attempt.device, dtype: attempt.dtype, scale: model.scale, mode, tileSize: model.tileSize, stride: model.stride, ring: model.ring, pad: model.pad, preClean: usePreClean ? { modelId: MODELS.denoise.id, bgr: false } : undefined, chain: mode === "logo" ? (targetScale >= 3 ? 2 : 1) : 1, targetScale: mode === "enhance" ? undefined : targetScale },
            (pct, tile, total) => {
              setStage("processing");
              if (tile && total) setTileInfo(isEs ? `Paso ${tile} de ${total}` : `Step ${tile} of ${total}`);
              setProgress(pct === 100 && !total ? 100 : pct);
            },
          );
          const secs = Math.round((performance.now() - t0) / 1000);
          setElapsed(secs >= 60 ? `${Math.floor(secs / 60)}m ${secs % 60}s` : `${secs}s`);

          const canvas = document.createElement("canvas");
          canvas.width = result.width;
          canvas.height = result.height;
          const ctx = canvas.getContext("2d")!;
          const img = ctx.createImageData(result.width, result.height);
          img.data.set(result.data);
          ctx.putImageData(img, 0, 0);
          const outBlob = await new Promise<Blob>((res) => canvas.toBlob((b) => res(b!), "image/png"));
          setResultUrl(URL.createObjectURL(outBlob));
          setResultDims({ w: result.width, h: result.height });
          setUsedLabel(`${(result.device ?? attempt.device) === "webgpu" ? "GPU" : "CPU"}`);
          setResultPre(!!result.pre);
          setStage("done");
          setDivider(50);
          return;
        } catch (err) {
          lastError = err instanceof Error ? err.message : String(err);
          if (lastError === "gpu-midrun-fail") {
            setGpuDied(true);
            setError(
              isEs
                ? "Tu gráfica se quedó sin memoria a mitad del proceso. Puedes terminarlo en CPU (puede tardar bastante) o probar con una imagen más pequeña."
                : "Your GPU ran out of memory mid-process. You can finish on CPU (it may take a while) or try a smaller image."
            );
            setStage("error");
            return;
          }
          spawnWorker();
        }
      }

      setError(
        lastError.includes("memory") || lastError.includes("allocation")
          ? isEs
            ? "La imagen agotó la memoria disponible. Prueba con una más pequeña."
            : "The image ran out of memory. Try a smaller one."
          : isEs
            ? `No se pudo procesar la imagen (${lastError}).`
            : `Could not process the image (${lastError}).`
      );
      setStage("error");
    },
    [model, mode, isEs, hasGpu, forceCpu, spawnWorker]
  );

  const onFile = useCallback(
    (file: File | undefined) => {
      if (!file || !file.type.startsWith("image/")) return;
      setOrigUrl("");
      setForceCpu(false);
      void run(file);
    },
    [run]
  );

  const download = useCallback(() => {
    if (!resultUrl) return;
    const prefix = mode === "enhance" ? "mejorada" : `ampliada-x${targetScale}`;
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = `${prefix}-${resultDims.w}x${resultDims.h}.png`;
    a.click();
  }, [resultUrl, resultDims, mode, targetScale]);

  const reset = useCallback(() => {
    setStage("idle");
    setOrigUrl("");
    setResultUrl("");
    setError("");
    setProgress(0);
    setTileInfo("");
    setGpuDied(false);
  }, []);

  const pointerMove = useCallback((clientX: number) => {
    const el = compareRef.current;
    if (!el || !draggingRef.current) return;
    const rect = el.getBoundingClientRect();
    setDivider(Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100)));
  }, []);

  useEffect(() => {
    const move = (e: PointerEvent) => pointerMove(e.clientX);
    const up = () => { draggingRef.current = false; };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [pointerMove]);

  const busy = stage === "loadingModel" || stage === "processing";

  return (
    <div className="space-y-5">
      {isMobile && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-300/90">
          <FiAlertCircle className="mt-0.5 shrink-0" />
          {isEs
            ? "Estos procesos necesitan un ordenador con suficiente memoria. En móvil funcionan solo con imágenes pequeñas."
            : "These processes need a computer with enough memory. On mobile they only work with small images."}
        </div>
      )}

      <div className="space-y-4 rounded-xl border border-border/20 bg-surface/30 p-4">
        <div>
          <label className="mb-2 block text-sm font-medium text-text">{isEs ? "1. ¿Qué quieres hacer?" : "1. What do you want to do?"}</label>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            {([
              ["upscale", FiMaximize2, isEs ? "Ampliar foto ×4" : "Enlarge photo ×4", isEs ? "Fotos hechas con cámara, ×4 más grandes" : "Camera photos, ×4 bigger"],
              ["logo", FiMaximize2, isEs ? "Ampliar logo o captura ×4" : "Enlarge logo or screenshot ×4", isEs ? "Logos, dibujos, capturas: motor de gráficos en 2 pasos" : "Logos, drawings, screenshots: graphics engine in 2 steps"],
              ["enhance", FiFileText, isEs ? "Mejorar mi foto" : "Enhance my photo", isEs ? "Mismo tamaño, mejor calidad" : "Same size, better quality"],
            ] as const).map(([id, Icon, title, desc]) => (
              <button key={id} onClick={() => setMode(id)} disabled={busy} className={`flex items-start gap-3 rounded-xl border p-3.5 text-left transition-colors disabled:opacity-40 ${mode === id ? "border-primary/60 bg-primary/10" : "border-border/30 bg-surface/40 hover:border-primary/30"}`}>
                <Icon className={`mt-0.5 text-lg ${mode === id ? "text-primary" : "text-text-muted"}`} />
                <span>
                  <span className={`block text-sm font-semibold ${mode === id ? "text-primary" : "text-text"}`}>{title}</span>
                  <span className="block text-xs text-text-muted/70">{desc}</span>
                </span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="mb-2 block text-sm font-medium text-text">{isEs ? "2. Cuéntame tu foto" : "2. Tell me about your photo"}</label>
          <div className="space-y-2">
            {availableKeys.map((k) => {
              const m = MODELS[k];
              const active = modelKey === k;
              return (
                <button key={k} onClick={() => setModelKey(k)} disabled={busy} className={`flex w-full items-start gap-3 rounded-xl border p-3 text-left transition-colors disabled:opacity-40 ${active ? "border-primary/60 bg-primary/10" : "border-border/30 bg-surface/40 hover:border-primary/30"}`}>
                  <span className={`mt-1 h-4 w-4 shrink-0 rounded-full border-2 ${active ? "border-primary bg-primary" : "border-text-muted/40"}`} />
                  <span>
                    <span className={`block text-sm font-semibold ${active ? "text-primary" : "text-text"}`}>{isEs ? m.es : m.en}</span>
                    <span className="block text-xs text-text-muted/70">{isEs ? m.descEs : m.descEn}</span>
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {mode !== "enhance" && (
          <div>
            <label className="mb-2 block text-sm font-medium text-text">{isEs ? "3. ¿Cuánto la quieres de grande?" : "3. How big should it be?"}</label>
            <div className="grid grid-cols-3 gap-2">
              {([2, 3, 4] as const).map((s) => (
                <button key={s} onClick={() => setTargetScale(s)} disabled={busy} className={`rounded-xl border p-3 text-center transition-colors disabled:opacity-40 ${targetScale === s ? "border-primary/60 bg-primary/10" : "border-border/30 bg-surface/40 hover:border-primary/30"}`}>
                  <span className={`block text-sm font-bold ${targetScale === s ? "text-primary" : "text-text"}`}>×{s}</span>
                  <span className="block text-[11px] text-text-muted/70">{s === 2 ? (isEs ? "Máxima nitidez" : "Max sharpness") : s === 3 ? (isEs ? "Equilibrio" : "Balanced") : isEs ? "Máximo tamaño" : "Max size"}</span>
                </button>
              ))}
            </div>
            <p className="mt-1.5 text-[11px] leading-relaxed text-text-muted/60">
              {isEs
                ? "El modelo siempre trabaja a ×4 internamente y ajusta al final: en ×2 las imperfecciones se suavizan y tarda lo mismo."
                : "The model always works at ×4 internally and adjusts at the end: ×2 smooths imperfections and takes the same time."}
            </p>
          </div>
        )}

        {!origUrl && (
          <label
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); onFile(e.dataTransfer.files?.[0]); }}
            className={`flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed p-10 text-center transition-colors ${dragOver ? "border-primary/60 bg-primary/5" : "border-border/30 hover:border-primary/40"}`}
          >
            <FiUploadCloud className="text-3xl text-text-muted/60" />
            <span className="text-sm font-semibold text-text">{isEs ? "4. Suelta tu foto aquí" : "4. Drop your photo here"}</span>
            <span className="text-xs text-text-muted/60">JPG · PNG · WebP — {isEs ? "se procesa en tu navegador, nunca se sube a internet" : "processed in your browser, never uploaded"}</span>
            <input type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
          </label>
        )}
      </div>

      {busy && (
        <div className="space-y-3 rounded-xl border border-border/20 bg-surface/30 p-6 text-center">
          <FiLoader className="mx-auto animate-spin text-2xl text-primary" />
          <p className="text-sm font-medium text-text">
            {stage === "loadingModel"
              ? isEs ? "Preparando herramientas de IA (solo la primera vez)…" : "Preparing AI tools (first time only)…"
              : isEs ? "Trabajando en tu foto…" : "Working on your photo…"}
          </p>
          <div className="mx-auto h-2 w-full max-w-md overflow-hidden rounded-full bg-white/10">
            <div className={`h-full rounded-full bg-primary transition-all duration-300 ${stage === "processing" ? "animate-pulse" : ""}`} style={{ width: `${stage === "loadingModel" ? Math.max(progress, 4) : Math.max(progress, 8)}%` }} />
          </div>
          <p className="flex items-center justify-center gap-1.5 text-xs text-text-muted/70"><FiClock /> {tileInfo || (isEs ? "Preparando…" : "Preparing…")}</p>
          <button onClick={() => { cancelRef.current = true; spawnWorker(); setStage("idle"); setProgress(0); setTileInfo(""); }} className="mx-auto block rounded-lg border border-border/30 bg-surface/60 px-4 py-1.5 text-xs font-medium text-text-muted transition-colors hover:text-text">
            {isEs ? "Cancelar" : "Cancel"}
          </button>
        </div>
      )}

      {error && (
        <div className="space-y-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          <p className="flex items-start gap-2"><FiAlertCircle className="mt-0.5 shrink-0" /> {error}</p>
          <div className="flex flex-wrap gap-2">
            {gpuDied && fileRef.current && (
              <button onClick={() => { setForceCpu(true); const f = fileRef.current; if (f) void run(f); }} className="flex items-center gap-1.5 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-1.5 text-xs font-semibold text-amber-300 transition-colors hover:bg-amber-500/20">
                <FiCpu /> {isEs ? "Continuar en CPU (lento)" : "Continue on CPU (slow)"}
              </button>
            )}
            {fileRef.current && (
              <button onClick={() => { const f = fileRef.current; reset(); if (f) onFile(f); }} className="rounded-lg border border-red-400/40 px-3 py-1.5 text-xs font-semibold text-red-300 transition-colors hover:bg-red-500/20">
                {isEs ? "Reintentar" : "Retry"}
              </button>
            )}
            <button onClick={reset} className="rounded-lg border border-border/30 px-3 py-1.5 text-xs font-medium text-text-muted transition-colors hover:text-text">
              {isEs ? "Empezar de nuevo" : "Start over"}
            </button>
          </div>
        </div>
      )}

      {stage === "done" && origUrl && resultUrl && (
        <>
          <div className="overflow-hidden rounded-xl border border-border/20">
            <div
              ref={compareRef}
              className="relative select-none touch-none"
              onPointerDown={(e) => { draggingRef.current = true; pointerMove(e.clientX); }}
              style={{ cursor: "ew-resize" }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={resultUrl} alt={isEs ? "Resultado" : "Result"} className="block w-full" draggable={false} />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={origUrl}
                alt={isEs ? "Original" : "Original"}
                draggable={false}
                className="absolute inset-0 block h-full w-full object-contain"
                style={{ clipPath: `inset(0 ${100 - divider}% 0 0)` }}
              />
              <div className="absolute inset-y-0 w-0.5 bg-primary shadow-[0_0_12px_rgba(124,58,237,0.8)]" style={{ left: `${divider}%` }}>
                <div className="absolute top-1/2 left-1/2 h-9 w-9 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary bg-black/70" />
              </div>
              <span className="absolute top-3 left-3 rounded bg-black/60 px-2 py-1 text-xs font-medium text-white">
                {isEs ? "Antes" : "Before"} · {origDims.w}×{origDims.h}
              </span>
              <span className="absolute top-3 right-3 rounded bg-primary/80 px-2 py-1 text-xs font-bold text-white">
                {isEs ? "Después" : "After"} · {resultDims.w}×{resultDims.h}
              </span>
            </div>
          </div>

          <p className="text-center text-xs text-text-muted/50">{isEs ? "Arrastra la línea para comparar el antes y el después." : "Drag the line to compare before and after."}</p>

          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border/20 bg-surface/30 p-4">
            <div className="text-xs text-text-muted">
              <p className="flex items-center gap-1.5"><FiZap className="text-primary" /> {usedLabel}{mode !== "enhance" ? ` · ×${targetScale}` : ""}{resultPre ? (isEs ? " · restauración previa aplicada (foto muy pequeña)" : " · pre-restoration applied (very small photo)") : ""}</p>
              <p className="flex items-center gap-1.5 mt-1"><FiClock /> {isEs ? "Tiempo" : "Time"}: {elapsed}</p>
              {capped && <p className="flex items-center gap-1.5 mt-1"><FiCpu /> {isEs ? `Tu foto original es de ${srcDims.w}×${srcDims.h} y se ajustó a ${origDims.w}×${origDims.h} antes de procesar.` : `Your original photo is ${srcDims.w}×${srcDims.h} and was resized to ${origDims.w}×${origDims.h} before processing.`}</p>}
            </div>
            <div className="flex gap-2">
              <button onClick={reset} className="flex items-center gap-1.5 rounded-lg border border-border/30 bg-surface/60 px-3 py-2 text-xs font-medium text-text-muted transition-colors hover:text-text"><FiX /> {isEs ? "Otra foto" : "Another photo"}</button>
              <button onClick={download} className="flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white transition-opacity hover:opacity-90"><FiDownload /> {isEs ? "Descargar" : "Download"} PNG · {resultDims.w}×{resultDims.h}</button>
            </div>
          </div>
        </>
      )}

      <details className="rounded-xl border border-border/20 bg-surface/30 p-4">
        <summary className="cursor-pointer text-xs font-medium text-text-muted/70">{isEs ? "Detalles técnicos" : "Technical details"}</summary>
        <div className="mt-3 space-y-1.5 text-xs leading-relaxed text-text-muted/60">
          <p>{isEs ? "Modelo actual:" : "Current model:"} <span className="font-mono">{model.id.split("/").pop()}</span> ({model.mb}MB)</p>
          <p>{isEs ? "Motor:" : "Engine:"} {model.engine === "raw" ? "ONNX Runtime Web" : "transformers.js"} · {hasGpu ? (isEs ? "WebGPU disponible" : "WebGPU available") : isEs ? "solo CPU" : "CPU only"}</p>
          <p>{isEs ? "Los modelos proceden de la comunidad de superresolución (NomosWebPhoto por Phips, Swin2SR por Microsoft Research) convertidos a ONNX." : "Models come from the super-resolution community (NomosWebPhoto by Phips, Swin2SR by Microsoft Research) converted to ONNX."}</p>
        </div>
      </details>
    </div>
  );
}
