import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
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
      "/runbooks": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000",
      "/tools": "http://127.0.0.1:8000",
      "/workspace": "http://127.0.0.1:8000"
    }
  },
  build: {
    outDir: "../static/dist",
    emptyOutDir: true
  }
});
