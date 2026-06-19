import "./tailwind.css";

import React from "react";
import ReactDOM from "react-dom/client";
import { Toaster } from "sonner";
import { ShadcnSmoke } from "./components/dev/ShadcnSmoke";
import { AppNext } from "./next/AppNext";

const search =
  typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;

// Dev-only: open with `?shadcn-smoke` to render the shadcn/ui smoke screen.
const showShadcnSmoke = search?.has("shadcn-smoke") ?? false;

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {showShadcnSmoke ? (
      <>
        <Toaster position="top-right" theme="dark" richColors />
        <ShadcnSmoke />
      </>
    ) : (
      <>
        <Toaster position="top-right" theme="dark" richColors />
        <AppNext />
      </>
    )}
  </React.StrictMode>
);
