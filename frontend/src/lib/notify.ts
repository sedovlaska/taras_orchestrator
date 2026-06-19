import { toast } from "sonner";

/**
 * Shared toast helpers backed by sonner.
 *
 * Migrated from `@mantine/notifications`. Both the current Mantine shell and
 * future shadcn/ui screens import these so toasts stay consistent across the
 * migration. The <Toaster/> is mounted once in main.tsx.
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
