/// <reference types="vite/client" />

declare module 'react-katex' {
  import React from 'react';
  export const InlineMath: React.FC<{ math: string }>;
  export const BlockMath: React.FC<{ math: string }>;
}
