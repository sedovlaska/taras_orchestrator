import "@mantine/core/styles.css";
import "./tailwind.css";
import "./styles.css";

import { MantineProvider, createTheme } from "@mantine/core";
import React from "react";
import ReactDOM from "react-dom/client";
import { Toaster } from "sonner";
import { App } from "./App";
import { ShadcnSmoke } from "./components/dev/ShadcnSmoke";
import { AppNext } from "./next/AppNext";

const search =
  typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;

// Dev-only: open with `?shadcn-smoke` to render the shadcn/ui smoke screen
// instead of the Mantine app. Never shown in the normal product flow.
const showShadcnSmoke = search?.has("shadcn-smoke") ?? false;

// Migration Phase 3+: open with `?ui=next` to render the new shadcn + AI SDK
// Elements chat surface. Flag-gated so the Mantine app stays the default shell
// until Phase 5 makes this view default and removes Mantine.
const showNext = search?.get("ui") === "next";

const theme = createTheme({
  primaryColor: "blue",
  defaultRadius: "sm",
  fontFamily:
    "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
  headings: {
    fontWeight: "750"
  }
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {showNext ? (
      <>
        <Toaster position="top-right" theme="dark" richColors />
        <AppNext />
      </>
    ) : showShadcnSmoke ? (
      <>
        <Toaster position="top-right" theme="dark" richColors />
        <ShadcnSmoke />
      </>
    ) : (
      <MantineProvider theme={theme} defaultColorScheme="dark">
        <Toaster position="top-right" theme="dark" richColors />
        <App />
      </MantineProvider>
    )}
  </React.StrictMode>
);
