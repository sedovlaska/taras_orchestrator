import { useState } from "react";
import { Rocket } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
} from "@/components/ui/tooltip";

/**
 * Dev-only smoke screen for the shadcn/ui + Tailwind v4 toolchain.
 *
 * NOT part of the product UI. It is rendered only when the app is opened with
 * `?shadcn-smoke` in the URL (see main.tsx). Its sole purpose is to prove that
 * Tailwind utilities + shadcn primitives compile and render.
 */
export function ShadcnSmoke() {
  const [name, setName] = useState("");

  return (
    <div className="dark min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-xl space-y-6">
        <div className="flex items-center gap-2">
          <Rocket className="size-5" />
          <h1 className="text-2xl font-semibold tracking-tight">
            shadcn/ui smoke test
          </h1>
          <Badge variant="secondary">dev only</Badge>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Toolchain check</CardTitle>
            <CardDescription>
              Tailwind v4 + shadcn primitives are rendering.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Tabs defaultValue="form">
              <TabsList>
                <TabsTrigger value="form">Form</TabsTrigger>
                <TabsTrigger value="info">Info</TabsTrigger>
              </TabsList>
              <TabsContent value="form" className="space-y-3">
                <Input
                  placeholder="Type a name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                />
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        onClick={() =>
                          toast.success("sonner works", {
                            description: name || "no name entered"
                          })
                        }
                      >
                        Fire a toast
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Triggers a sonner toast</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </TabsContent>
              <TabsContent value="info">
                <p className="text-sm text-muted-foreground">
                  If this panel is styled, Tailwind compiled correctly.
                </p>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
