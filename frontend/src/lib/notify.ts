import { toast } from "sonner";

/**
 * Shared toast helpers backed by sonner.
 *
 * The <Toaster/> is mounted once in main.tsx.
 */

export function notifyError(error: unknown, title = "Request failed") {
  toast.error(title, {
    description: error instanceof Error ? error.message : String(error)
  });
}

export function notifySuccess(title: string, message?: string) {
  toast.success(title, message ? { description: message } : undefined);
}

export function notifyInfo(title: string, message?: string) {
  toast.info(title, message ? { description: message } : undefined);
}
