"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FiUploadCloud, FiDownload, FiLoader, FiX,
  FiAlertCircle, FiCpu, FiZap, FiInfo, FiClock,
} from "react-icons/fi";

interface Props { locale?: string; }

type Stage = "idle" | "loadingModel" | "processing" | "done" | "error";
type ModelKey = "x2-light" | "x2-classic" | "x4-real";
type Mode = "upscale" | "enhance";

const MODEL_DEFS: Record<ModelKey, {
  id: string; scale: 2 | 4; tileSize: number; pad: number; maxSide: number; shotLimit: number;
  es: string; en: string; descEs: string; descEn: string;
}> = {
  "x2-light": { id: "Xenova/swin2SR-lightweight-x2-64", scale: 2, tileSize: 256, pad: 32, maxSide: 2048, shotLimit: 1024, es: "Rápido ×2", en: "Fast ×2", descEs: "El más ligero y veloz. Ideal para capturas e imágenes pequeñas.", descEn: "Lightest and fastest. Great for screenshots and small images." },
  "x2-classic": { id: "Xenova/swin2SR-classical-sr-x2-64", scale: 2, tileSize: 256, pad: 32, maxSide: 1536, shotLimit: 800, es: "Calidad ×2", en: "Quality ×2", descEs: "Más detalle que el rápido, mismo factor ×2.", descEn: "More detail than fast, same ×2 factor." },
  "x4-real": { id: "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr", scale: 4, tileSize: 256, pad: 32, maxSide: 1280, shotLimit: 512, es: "Fotos reales ×4", en: "Real photos ×4", descEs: "Entrenado para fotografía real. El más lento: imágenes grandes van por teselas.", descEn: "Trained on real photography. Slowest: large images go tile by tile." },
};

interface Attempt { device: "webgpu" | "wasm"; dtype?: string; }
let reqCounter = 0;

interface WorkerResult { data: Uint8ClampedArray; width: number; height: number; }

function runInWorker(
  worker: Worker,
  payload: { bytes: ArrayBuffer; modelId: string; device: string; dtype?: string; scale: number; mode: Mode; tileSize: number; pad: number },
  onProgress: (pct: number, tile?: number, total?: number) => void,
  onProcessing: () => void,
): Promise<WorkerResult> {
  return new Promise((resolve, reject) => {
    const reqId = ++reqCounter;
    const buf = payload.bytes.slice(0);
    const handler = (e: MessageEvent) => {
      const m = e.data as { type: string; reqId: number; pct?: number; tile?: number; total?: number; data?: ArrayBuffer; width?: number; height?: number; message?: string };
      if (m.reqId !== reqId) return;
      if (m.type === "progress") onProgress(m.pct ?? 0, m.tile, m.total);
      else if (m.type === "processing") onProcessing();
      else if (m.type === "done") {
        worker.removeEventListener("message", handler);
        resolve({ data: new Uint8ClampedArray(m.data!), width: m.width!, height: m.height! });
      } else if (m.type === "error") {
        worker.removeEventListener("message", handler);
        reject(new Error(m.message ?? "error"));
      }
    };
    worker.addEventListener("message", handler);
    try {
      worker.postMessage({ type: "run", reqId, bytes: buf, modelId: payload.modelId, device: payload.device, dtype: payload.dtype, scale: payload.scale, mode: payload.mode, tileSize: payload.tileSize, pad: payload.pad }, [buf]);
    } catch (err) {
      worker.removeEventListener("message", handler);
      reject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

async function downscaleToCap(file: File, maxSide: number): Promise<{ bytes: ArrayBuffer; url: string; width: number; height: number; capped: boolean }> {
  const bmp = await createImageBitmap(file);
  const w = bmp.width;
  const h = bmp.height;
  const scale = Math.min(1, maxSide / Math.max(w, h));
  if (scale >= 1) {
    const buf = await file.arrayBuffer();
    bmp.close();
    return { bytes: buf, url: URL.createObjectURL(file), width: w, height: h, capped: false };
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
  return { bytes: buf, url: URL.createObjectURL(blob), width: nw, height: nh, capped: true };
}

export default function ImageUpscaler({ locale = "es" }: Props) {
  const isEs = locale === "es";

  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [tileInfo, setTileInfo] = useState("");
  const [error, setError] = useState("");
  const [modelKey, setModelKey] = useState<ModelKey>("x2-light");
  const [mode, setMode] = useState<Mode>("upscale");
  const [origUrl, setOrigUrl] = useState("");
  const [resultUrl, setResultUrl] = useState("");
  const [origDims, setOrigDims] = useState({ w: 0, h: 0 });
  const [resultDims, setResultDims] = useState({ w: 0, h: 0 });
  const [usedLabel, setUsedLabel] = useState("");
  const [elapsed, setElapsed] = useState("");
  const [capped, setCapped] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [divider, setDivider] = useState(50);

  const workerRef = useRef<Worker | null>(null);
  const fileRef = useRef<File | null>(null);
  const compareRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  useEffect(() => {
    setIsMobile(window.matchMedia("(max-width: 639px)").matches);
    const w = new Worker(new URL("./imageUpscale.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = w;
    return () => w.terminate();
  }, []);

  const model = MODEL_DEFS[modelKey];

  useEffect(() => {
    setStage((s) => (s === "error" ? "idle" : s));
    setError("");
    setProgress(0);
  }, [modelKey, mode]);

  const run = useCallback(
    async (file: File) => {
      setStage("loadingModel");
      setProgress(0);
      setTileInfo("");
      setError("");
      setResultUrl("");

      const prepared = await downscaleToCap(file, model.maxSide);
      setOrigUrl(prepared.url);
      setOrigDims({ w: prepared.width, h: prepared.height });
      setCapped(prepared.capped);
      fileRef.current = file;

      const singleShot = Math.max(prepared.width, prepared.height) <= model.shotLimit;

      const attempts: Attempt[] = [
        { device: "wasm", dtype: "q8" },
        { device: "webgpu", dtype: "fp16" },
      ];

      let lastError = "";
      for (const attempt of attempts) {
        try {
          const hasGpu = typeof navigator !== "undefined" && "gpu" in navigator;
          if (attempt.device === "webgpu" && !hasGpu) continue;
          const worker = workerRef.current;
          if (!worker) throw new Error("no-worker");

          const t0 = performance.now();
          const result = await runInWorker(
            worker,
            { bytes: prepared.bytes, modelId: model.id, device: attempt.device, dtype: attempt.dtype, scale: model.scale, mode, tileSize: singleShot ? 0 : model.tileSize, pad: model.pad },
            (pct, tile, total) => {
              setStage("processing");
              if (tile && total) setTileInfo(singleShot ? (isEs ? "Procesando imagen completa…" : "Processing full image…") : isEs ? `Tesela ${tile} de ${total}` : `Tile ${tile} of ${total}`);
              setProgress(pct === 100 && !total ? 100 : pct);
            },
            () => setStage("processing"),
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
          setUsedLabel(`${attempt.device === "webgpu" ? "GPU" : "CPU"} · ${mode === "enhance" ? (isEs ? "mejora a resolución original" : "enhance at original size") : `${model.scale}×`}`);
          setStage("done");
          setDivider(50);
          return;
        } catch (err) {
          lastError = err instanceof Error ? err.message : String(err);
        }
      }

      setError(
        lastError.includes("memory") || lastError.includes("allocation")
          ? isEs
            ? "La imagen agotó la memoria disponible. Prueba el modelo rápido ×2 o reduce el tamaño."
            : "The image ran out of memory. Try the fast ×2 model or reduce its size."
          : isEs
            ? `No se pudo procesar la imagen (${lastError}).`
            : `Could not process the image (${lastError}).`
      );
      setStage("error");
    },
    [model, mode, isEs]
  );

  const onFile = useCallback(
    (file: File | undefined) => {
      if (!file || !file.type.startsWith("image/")) return;
      setOrigUrl("");
      void run(file);
    },
    [run]
  );

  const download = useCallback(() => {
    if (!resultUrl) return;
    const prefix = mode === "enhance" ? "mejorada" : `ampliada-x${model.scale}`;
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = `${prefix}-${resultDims.w}x${resultDims.h}.png`;
    a.click();
  }, [resultUrl, resultDims, mode, model.scale]);

  const reset = useCallback(() => {
    setStage("idle");
    setOrigUrl("");
    setResultUrl("");
    setError("");
    setProgress(0);
    setTileInfo("");
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
            ? "La superresolución con IA necesita bastante RAM y CPU: en móvil funciona solo con imágenes pequeñas. Para ×4 recomendamos escritorio."
            : "AI super-resolution needs plenty of RAM and CPU: on mobile it only works with small images. For ×4 we recommend desktop."}
        </div>
      )}

      <div className="space-y-4 rounded-xl border border-border/20 bg-surface/30 p-4">
        <div>
          <label className="mb-2 block text-xs text-text-muted/70">{isEs ? "¿Qué quieres hacer?" : "What do you want to do?"}</label>
          <div className="flex flex-wrap gap-2">
            {([
              ["upscale", isEs ? "Ampliar resolución" : "Upscale resolution"],
              ["enhance", isEs ? "Solo mejorar calidad" : "Enhance only"],
            ] as const).map(([id, label]) => (
              <button key={id} onClick={() => setMode(id)} disabled={busy} className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-40 ${mode === id ? "border-primary/50 bg-primary/10 text-primary" : "border-border/30 bg-surface/60 text-text-muted hover:text-text"}`}>{label}</button>
            ))}
          </div>
          <p className="mt-1.5 text-xs text-text-muted/50">
            {mode === "upscale"
              ? isEs ? "Multiplica la resolución (×2 o ×4 según el modelo)." : "Multiplies resolution (×2 or ×4 depending on model)."
              : isEs ? "Mantiene tu resolución actual y aplica el modelo para limpiar y detallar." : "Keeps your current resolution while applying the model to clean and detail."}
          </p>
        </div>

        <div>
          <label className="mb-2 block text-xs text-text-muted/70">{isEs ? "Modelo" : "Model"}</label>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(MODEL_DEFS) as ModelKey[]).map((k) => (
              <button
                key={k}
                onClick={() => setModelKey(k)}
                disabled={busy}
                className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-40 ${modelKey === k ? "border-primary/50 bg-primary/10 text-primary" : "border-border/30 bg-surface/60 text-text-muted hover:text-text"}`}
              >
                {isEs ? MODEL_DEFS[k].es : MODEL_DEFS[k].en}
              </button>
            ))}
          </div>
          <p className="mt-2 flex items-start gap-1.5 text-xs text-text-muted/60">
            <FiInfo className="mt-0.5 shrink-0" />
            {isEs ? model.descEs : model.descEn}{" "}
            {isEs ? `Máximo por lado: ${model.maxSide}px.` : `Max side: ${model.maxSide}px.`}
          </p>
        </div>

        {!origUrl && (
          <label
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); onFile(e.dataTransfer.files?.[0]); }}
            className={`flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed p-10 text-center transition-colors ${dragOver ? "border-primary/60 bg-primary/5" : "border-border/30 hover:border-primary/40"}`}
          >
            <FiUploadCloud className="text-3xl text-text-muted/60" />
            <span className="text-sm font-medium text-text">{isEs ? "Suelta una imagen o haz clic" : "Drop an image or click"}</span>
            <span className="text-xs text-text-muted/60">JPG · PNG · WebP — {isEs ? "se procesa en tu navegador, nunca se sube" : "processed in your browser, never uploaded"}</span>
            <input type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
          </label>
        )}
      </div>

      {busy && (
        <div className="space-y-3 rounded-xl border border-border/20 bg-surface/30 p-6 text-center">
          <FiLoader className="mx-auto animate-spin text-2xl text-primary" />
          <p className="text-sm font-medium text-text">
            {stage === "loadingModel"
              ? isEs ? "Descargando modelo IA (primera vez; luego queda en caché)…" : "Downloading AI model (first time; cached afterwards)…"
              : isEs ? "Procesando… en CPU puede tardar desde segundos hasta varios minutos según tamaño" : "Processing… on CPU this can take seconds to several minutes depending on size"}
          </p>
          <div className="mx-auto h-2 w-full max-w-md overflow-hidden rounded-full bg-white/10">
            <div className={`h-full rounded-full bg-primary transition-all duration-300 ${stage === "processing" ? "animate-pulse" : ""}`} style={{ width: `${stage === "loadingModel" ? Math.max(progress, 4) : Math.max(progress, 8)}%` }} />
          </div>
          <p className="flex items-center justify-center gap-1.5 text-xs text-text-muted/70"><FiClock /> {tileInfo || (isEs ? "Preparando…" : "Preparing…")}</p>
        </div>
      )}

      {error && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">
          <FiAlertCircle className="shrink-0" /> {error}
          {fileRef.current && (
            <button onClick={() => fileRef.current && void run(fileRef.current)} className="ml-auto rounded-lg border border-red-400/40 px-3 py-1.5 text-xs font-semibold text-red-300 transition-colors hover:bg-red-500/20">
              {isEs ? "Reintentar" : "Retry"}
            </button>
          )}
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
                {isEs ? "Original" : "Original"} · {origDims.w}×{origDims.h}
              </span>
              <span className="absolute top-3 right-3 rounded bg-primary/80 px-2 py-1 text-xs font-bold text-white">
                {mode === "enhance" ? (isEs ? "Mejorada" : "Enhanced") : "IA"} · {resultDims.w}×{resultDims.h}
              </span>
            </div>
          </div>

          <p className="text-center text-xs text-text-muted/50">{isEs ? "Arrastra el divisor para comparar antes y después." : "Drag the divider to compare before and after."}</p>

          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border/20 bg-surface/30 p-4">
            <div className="text-xs text-text-muted">
              <p className="flex items-center gap-1.5"><FiZap className="text-primary" /> {usedLabel}</p>
              <p className="flex items-center gap-1.5 mt-1"><FiClock /> {isEs ? "Tiempo de proceso" : "Process time"}: {elapsed}</p>
              {capped && <p className="flex items-center gap-1.5 mt-1"><FiCpu /> {isEs ? "Entrada reducida al límite del modelo antes de procesar." : "Input downscaled to the model limit before processing."}</p>}
            </div>
            <div className="flex gap-2">
              <button onClick={reset} className="flex items-center gap-1.5 rounded-lg border border-border/30 bg-surface/60 px-3 py-2 text-xs font-medium text-text-muted transition-colors hover:text-text"><FiX /> {isEs ? "Nueva imagen" : "New image"}</button>
              <button onClick={download} className="flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white transition-opacity hover:opacity-90"><FiDownload /> PNG · {resultDims.w}×{resultDims.h}</button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
