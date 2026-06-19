import "@mantine/core/styles.css";
import "./tailwind.css";
import "./styles.css";

import { MantineProvider, createTheme } from "@mantine/core";
import React from "react";
import ReactDOM from "react-dom/client";
import { Toaster } from "sonner";
import { App } from "./App";
import { ShadcnSmoke } from "./components/dev/ShadcnSmoke";

// Dev-only: open with `?shadcn-smoke` to render the shadcn/ui smoke screen
// instead of the Mantine app. Never shown in the normal product flow.
const showShadcnSmoke =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).has("shadcn-smoke");

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
    {showShadcnSmoke ? (
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
