"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  FiUploadCloud, FiDownload, FiLoader, FiX,
  FiAlertCircle, FiCpu, FiZap, FiInfo,
} from "react-icons/fi";

interface Props { locale?: string; }

type Stage = "idle" | "loadingModel" | "processing" | "done" | "error";
type ModelKey = "x2-light" | "x2-classic" | "x4-real";

const MODELS: Record<ModelKey, { id: string; scale: 2 | 4; es: string; en: string; descEs: string; descEn: string; maxSide: number }> = {
  "x2-light": { id: "Xenova/swin2SR-lightweight-x2-64", scale: 2, es: "Rápido ×2", en: "Fast ×2", descEs: "El más ligero. Ideal para pantallas y fotos pequeñas.", descEn: "Lightest one. Great for screens and small photos.", maxSide: 1024 },
  "x2-classic": { id: "Xenova/swin2SR-classical-sr-x2-64", scale: 2, es: "Calidad ×2", en: "Quality ×2", descEs: "Más detalle que el rápido, mismo factor ×2.", descEn: "More detail than fast, same ×2 factor.", maxSide: 1024 },
  "x4-real": { id: "Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr", scale: 4, es: "Fotos reales ×4", en: "Real photos ×4", descEs: "Entrenado para fotos de la vida real (BSRGAN). El más exigente.", descEn: "Trained on real-world photos (BSRGAN). Most demanding.", maxSide: 768 },
};

interface Attempt { device: "webgpu" | "wasm"; dtype?: string; }
let reqCounter = 0;

interface WorkerResult { data: Uint8ClampedArray; width: number; height: number; }

function runInWorker(
  worker: Worker,
  payload: { bytes: ArrayBuffer; modelId: string; device: string; dtype?: string },
  onProgress: (pct: number) => void,
  onProcessing: () => void,
): Promise<WorkerResult> {
  return new Promise((resolve, reject) => {
    const reqId = ++reqCounter;
    const handler = (e: MessageEvent) => {
      const m = e.data as { type: string; reqId: number; pct?: number; data?: ArrayBuffer; width?: number; height?: number; message?: string };
      if (m.reqId !== reqId) return;
      if (m.type === "progress") onProgress(m.pct ?? 0);
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
    worker.postMessage({ type: "run", reqId, bytes: payload.bytes, modelId: payload.modelId, device: payload.device, dtype: payload.dtype }, [payload.bytes]);
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
  const [error, setError] = useState("");
  const [modelKey, setModelKey] = useState<ModelKey>("x2-light");
  const [origUrl, setOrigUrl] = useState("");
  const [resultUrl, setResultUrl] = useState("");
  const [origDims, setOrigDims] = useState({ w: 0, h: 0 });
  const [resultDims, setResultDims] = useState({ w: 0, h: 0 });
  const [usedLabel, setUsedLabel] = useState("");
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

  const model = MODELS[modelKey];

  const run = useCallback(
    async (file: File) => {
      setStage("loadingModel");
      setProgress(0);
      setError("");
      setResultUrl("");
      setUsedLabel("");

      const prepared = await downscaleToCap(file, model.maxSide);
      setOrigUrl(prepared.url);
      setOrigDims({ w: prepared.width, h: prepared.height });
      setCapped(prepared.capped);
      fileRef.current = file;

      const attempts: Attempt[] = [
        { device: "webgpu", dtype: "fp16" },
        { device: "wasm", dtype: "q8" },
      ];

      let lastError = "";
      for (const attempt of attempts) {
        try {
          const hasGpu = typeof navigator !== "undefined" && "gpu" in navigator;
          if (attempt.device === "webgpu" && !hasGpu) continue;
          const worker = workerRef.current;
          if (!worker) throw new Error("no-worker");

          const result = await runInWorker(
            worker,
            { bytes: prepared.bytes, modelId: model.id, device: attempt.device, dtype: attempt.dtype },
            setProgress,
            () => setStage("processing"),
          );

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
          setUsedLabel(`${attempt.device === "webgpu" ? "GPU" : "CPU"} · ${model.scale}×`);
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
            ? "La imagen es demasiado grande para la memoria disponible. Prueba con una más pequeña o el modelo rápido ×2."
            : "The image is too large for available memory. Try a smaller one or the fast ×2 model."
          : isEs
            ? `No se pudo procesar la imagen (${lastError}).`
            : `Could not process the image (${lastError}).`
      );
      setStage("error");
    },
    [model, isEs]
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
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = `upscaler-${resultDims.w}x${resultDims.h}.png`;
    a.click();
  }, [resultUrl, resultDims]);

  const reset = useCallback(() => {
    setStage("idle");
    setOrigUrl("");
    setResultUrl("");
    setError("");
    setProgress(0);
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

  return (
    <div className="space-y-5">
      {isMobile && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs leading-relaxed text-amber-300/90">
          <FiAlertCircle className="mt-0.5 shrink-0" />
          {isEs
            ? "La superresolución con IA necesita bastante RAM y GPU: en móvil funciona solo con imágenes pequeñas. Para resultados ×4 recomendamos escritorio."
            : "AI super-resolution needs plenty of RAM and GPU: on mobile it only works with small images. For ×4 results we recommend desktop."}
        </div>
      )}

      <div className="space-y-4 rounded-xl border border-border/20 bg-surface/30 p-4">
        <div>
          <label className="mb-2 block text-xs text-text-muted/70">{isEs ? "Modelo" : "Model"}</label>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(MODELS) as ModelKey[]).map((k) => (
              <button
                key={k}
                onClick={() => setModelKey(k)}
                disabled={stage === "loadingModel" || stage === "processing"}
                className={`rounded-lg border px-3 py-2 text-sm font-medium transition-colors disabled:opacity-40 ${modelKey === k ? "border-primary/50 bg-primary/10 text-primary" : "border-border/30 bg-surface/60 text-text-muted hover:text-text"}`}
              >
                {isEs ? MODELS[k].es : MODELS[k].en}
              </button>
            ))}
          </div>
          <p className="mt-2 flex items-start gap-1.5 text-xs text-text-muted/60">
            <FiInfo className="mt-0.5 shrink-0" />
            {isEs ? model.descEs : model.descEn}{" "}
            {isEs ? `Máximo recomendado por lado: ${model.maxSide}px.` : `Recommended max side: ${model.maxSide}px.`}
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

      {(stage === "loadingModel" || stage === "processing") && (
        <div className="space-y-3 rounded-xl border border-border/20 bg-surface/30 p-6 text-center">
          <FiLoader className="mx-auto animate-spin text-2xl text-primary" />
          <p className="text-sm font-medium text-text">
            {stage === "loadingModel"
              ? isEs ? "Descargando modelo IA (primera vez; luego queda en caché)…" : "Downloading AI model (first time; cached afterwards)…"
              : isEs ? "Ampliando imagen… esto puede tardar unos segundos" : "Upscaling image… this may take a few seconds"}
          </p>
          <div className="mx-auto h-2 w-full max-w-md overflow-hidden rounded-full bg-white/10">
            <div className="h-full rounded-full bg-primary transition-all duration-300" style={{ width: `${stage === "processing" ? 100 : progress}%` }} />
          </div>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">
          <FiAlertCircle className="mt-0.5 shrink-0" /> {error}
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
              <img src={resultUrl} alt={isEs ? "Resultado ampliado" : "Upscaled result"} className="block w-full" draggable={false} />
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
                IA · {resultDims.w}×{resultDims.h}
              </span>
            </div>
          </div>

          <p className="text-center text-xs text-text-muted/50">{isEs ? "Arrastra el divisor para comparar antes y después." : "Drag the divider to compare before and after."}</p>

          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border/20 bg-surface/30 p-4">
            <div className="text-xs text-text-muted">
              <p className="flex items-center gap-1.5"><FiZap className="text-primary" /> {usedLabel}</p>
              {capped && <p className="flex items-center gap-1.5 mt-1"><FiCpu /> {isEs ? "Entrada reducida al límite del modelo antes de ampliar." : "Input downscaled to the model limit before upscaling."}</p>}
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
