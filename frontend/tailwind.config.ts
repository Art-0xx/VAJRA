import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#030712",
        foreground: "#f3f4f6",
        vajra: {
          dark: "#000000",
          panel: "#090d16",
          card: "#0d1322",
          border: "#1f293d",
          accent: "#38bdf8",
          emerald: "#10b981",
          ruby: "#ef4444",
          amber: "#f59e0b",
        }
      },
      fontFamily: {
        mono: ["var(--font-geist-mono)", "JetBrains Mono", "Courier New", "monospace"],
      }
    },
  },
  plugins: [],
};
export default config;
