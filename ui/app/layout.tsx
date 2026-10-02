import type { Metadata } from "next";
import React from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "DAVID-Net | Multimodal Deepfake Forensic Detection",
  description:
    "Disentangled Audio-Visual Deepfake Attribution, 4-Quadrant Classification, and Temporal Localization",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const theme = localStorage.getItem('david_theme');
                if (theme === 'dark') {
                  document.documentElement.classList.add('dark');
                } else {
                  document.documentElement.classList.remove('dark');
                }
              } catch (_) {}
            `,
          }}
        />
      </head>
      <body className="bg-[#F8FAFC] dark:bg-[#08090D] text-slate-900 dark:text-zinc-100 min-h-screen antialiased selection:bg-sky-500 selection:text-white transition-colors duration-200">
        {children}
      </body>
    </html>
  );
}
