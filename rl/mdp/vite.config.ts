import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Pure front-end app. `npm run dev` serves it; `npm run build` outputs static files.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
