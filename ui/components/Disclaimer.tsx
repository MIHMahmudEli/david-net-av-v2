"use client";

import React from "react";
import { AlertCircle } from "lucide-react";

export const Disclaimer: React.FC = () => {
  return (
    <div className="border border-amber-300/80 dark:border-amber-500/30 bg-amber-50/90 dark:bg-amber-950/20 backdrop-blur-md rounded-2xl p-4 sm:p-5 text-xs text-amber-900 dark:text-amber-200/90 flex items-start space-x-3 shadow-sm">
      <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
      <div>
        <strong className="font-bold text-amber-950 dark:text-amber-200 block mb-0.5">
          Probabilistic Decision Support Guarantee:
        </strong>
        <span className="text-amber-800/90 dark:text-amber-300/80 leading-relaxed block">
          Detection outputs are algorithmic inference estimates calibrated on
          multimodal deepfake corpora. They provide human-in-the-loop decision
          support for digital forensic analysts, fact-checkers, and media
          platforms, and do not constitute autonomous legal proof of tampering.
        </span>
      </div>
    </div>
  );
};
