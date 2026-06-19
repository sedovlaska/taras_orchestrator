import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./frontend/src", import.meta.url))
    }
  },
  root: "frontend",
  server: {
    proxy: {
      "/agents": "http://127.0.0.1:8000",
      "/approvals": "http://127.0.0.1:8000",
      "/chat": "http://127.0.0.1:8000",
      "/context": "http://127.0.0.1:8000",
      "/diagnostics": "http://127.0.0.1:8000",
      "/evals": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/models": "http://127.0.0.1:8000",
      "/runbooks": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000",
      "/settings": "http://127.0.0.1:8000",
      "/tools": "http://127.0.0.1:8000",
      "/conversations": "http://127.0.0.1:8000",
      "/workspace": "http://127.0.0.1:8000"
    }
  },
  build: {
    outDir: "../static/dist",
    emptyOutDir: true
  }
});
