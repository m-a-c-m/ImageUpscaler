# AI Image Upscaler — Free Online Tool

**AI Image Upscaler.** Upscale photos x2/x4 with Swin2SR neural networks running locally in your browser via WebGPU. No sign-up, no ads, 100% client-side.

🌐 **Demo en vivo / Live demo:** [miguelacm.es/tools/image-upscaler](https://miguelacm.es/tools/image-upscaler)

[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?logo=typescript)](https://www.typescriptlang.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4-06B6D4?logo=tailwindcss)](https://tailwindcss.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Features

- **🚀 Local AI super-resolution:** Swin2SR models run on your own GPU through WebGPU with automatic CPU fallback.
- **🖼️ Before/after slider:** Drag the divider to compare original and result in real time.
- **🔒 100% private:** Your image is processed on your device and never uploaded.

---

## 🚀 Quick start

```bash
git clone https://github.com/m-a-c-m/ImageUpscaler.git
cd ImageUpscaler
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Environment variables (optional)

```env
NEXT_PUBLIC_SITE_URL=https://miguelacm.es/tools/image-upscaler
NEXT_PUBLIC_EMBED_URL=https://miguelacm.es/embed/image-upscaler
```

---

## 📦 Embed on your website

```html
<iframe
  src="https://miguelacm.es/embed/image-upscaler"
  width="100%"
  height="700"
  style="border:none;border-radius:12px;"
  title="AI Image Upscaler — miguelacm.es"
  loading="lazy"
></iframe>
```

### Link with attribution (recommended for backlink)

```html
<a href="https://miguelacm.es/tools/image-upscaler" target="_blank" rel="noopener">
  AI Image Upscaler — free tool by MACM
</a>
```

> 💡 The link option generates a real backlink that benefits the project.

---

## 🛠 Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| [Next.js](https://nextjs.org) | 16 | React framework |
| [TypeScript](https://www.typescriptlang.org) | 5 | Type safety |
| [Tailwind CSS](https://tailwindcss.com) | 4 | Styling |
| [react-icons](https://react-icons.github.io/react-icons/) | 5 | Icons |
| `@huggingface/transformers` | — | Core logic |

---

## 📄 License

MIT © [Miguel Ángel Colorado Marin (MACM)](https://miguelacm.es)

Built with ❤️ by **[MACM](https://miguelacm.es)** — Full Stack Developer & Cybersecurity Specialist from Guadalajara, Spain.

- 🌐 Portfolio: [miguelacm.es](https://miguelacm.es)
- 💼 LinkedIn: [linkedin.com/in/macm](https://www.linkedin.com/in/macm/)
- 🐙 GitHub: [github.com/m-a-c-m](https://github.com/m-a-c-m)
